from os import environ
from os.path import join
from tempfile import gettempdir

# Wall-clock cap on a single ILP solve.
ILP_SOLVER_TIMEOUT = 600

# CBC prints a ~40-line report (version banner, cut statistics, timings) to the
# solver process' stdout for every single ILP solved, and pulp leaves that
# enabled by default. Inside an Apache fcgid worker that stdout is inherited
# straight from the parent, so mod_fcgid dumps it verbatim into Apache's
# error_log -- one report per molecule, tens of MB a week, burying the actual
# errors. Nothing in ATB ever reads it: solver failures surface as
# PulpSolverError / a non-optimal problem.status, both handled at the call sites.
#
# Set ATB_FRAGMENT_CAPPING_SOLVER_MSG=true to get the reports back when
# debugging a solver problem interactively.
#
# Relatedly: the LpProblem names in helpers/ use underscores rather than spaces
# on purpose. pulp rewrites a name containing a space and emits a UserWarning
# each time, which lands in the same place for the same reason. Please keep them
# space-free.
SOLVER_MSG = environ.get('ATB_FRAGMENT_CAPPING_SOLVER_MSG', 'false').lower() == 'true'

# When an ILP fails, the call sites can dump the model (a .lp file) and a graph
# of the molecule for post-mortem analysis. Those dumps are written relative to
# the process' working directory, which for an Apache fcgid worker is the
# website's DocumentRoot -- so in production they accumulated as world-readable
# litter inside the served tree (48 *_debug.lp files at the last count) rather
# than anywhere anyone would look for them.
#
# Off by default. Turn it on, per-process, only while chasing a solver failure:
#   ATB_FRAGMENT_CAPPING_DEBUG=true [ATB_FRAGMENT_CAPPING_DEBUG_DIR=/some/dir]
# The failure itself is unaffected either way -- the exception still propagates,
# with the molecule name and solver status in it.
WRITE_FAILED_ILP_DEBUG = environ.get('ATB_FRAGMENT_CAPPING_DEBUG', 'false').lower() == 'true'

# Never the working directory: see above.
FAILED_ILP_DEBUG_DIR = environ.get(
    'ATB_FRAGMENT_CAPPING_DEBUG_DIR',
    join(gettempdir(), 'fragment_capping_debug'),
)


def configure_default_solver() -> None:
    '''Apply SOLVER_MSG to the solver pulp uses when a call site passes none.

    Belt and braces with ilp_solver() below: a call site that passes no solver at
    all still gets a quiet one.
    '''
    try:
        from pulp import LpSolverDefault
    except ImportError:
        return
    if LpSolverDefault is not None:
        LpSolverDefault.msg = SOLVER_MSG


def _solver_log_tail(log_path: str, max_lines: int = 30) -> str:
    '''The last few lines of a CBC run's output, for an error message.'''
    try:
        with open(log_path) as file_handler:
            output = file_handler.read()
    except OSError as error:
        return '<could not read solver log {0}: {1}>'.format(log_path, error)

    if output.strip() == '':
        # CBC always prints a version banner before doing anything, so an empty
        # log means it died before or during start-up rather than failing on the
        # model -- killed by a signal, or unable to exec at all.
        return '<empty: CBC produced no output at all>'

    lines = output.splitlines()
    return '\n'.join(
        (['... ({0} earlier lines omitted)'.format(len(lines) - max_lines)] if len(lines) > max_lines else [])
        + lines[-max_lines:]
    )


def _diagnosable_cbc_class():
    '''PULP_CBC_CMD, subclassed so that a failed solve says what CBC actually did.

    pulp's solve_CBC checks only the CBC child's exit status and, when it is
    non-zero, raises ``PulpSolverError('Pulp: Error while trying to execute, use
    msg=True for more details' + path)`` -- having sent CBC's stdout and stderr to
    /dev/null. The one report that would explain the failure is therefore
    discarded at exactly the moment it is needed, and the advice in the message is
    no help after the fact: acting on it means catching the same failure a second
    time, which for a transient one may take days.

    So route CBC's output to a temp file (via pulp's own ``logPath`` option, which
    solve_CBC closes before raising) and put the tail of it in the exception.
    A CBC that failed on the model explains itself there; a CBC that was killed
    leaves the log empty or cut off mid-solve, which is itself the answer.

    Built lazily rather than at module scope so that importing this config module
    does not require pulp -- see configure_default_solver().
    '''
    from pulp import PULP_CBC_CMD, PulpSolverError

    class Diagnosable_PULP_CBC_CMD(PULP_CBC_CMD):
        def actualSolve(self, lp, **kwargs):
            if self.msg or self.optionsDict.get('logPath'):
                # Output is already going somewhere the caller can see.
                return super().actualSolve(lp, **kwargs)

            from os import close, remove
            from tempfile import mkstemp

            handle, log_path = mkstemp(prefix='cbc-', suffix='.log')
            close(handle)
            # Per-solve, and this object is only ever used by one thread at a time
            # (ilp_solver() hands out a fresh one; sequentialSolve reuses it in
            # sequence), so mutating optionsDict here is safe.
            self.optionsDict['logPath'] = log_path
            try:
                return super().actualSolve(lp, **kwargs)
            except PulpSolverError as error:
                raise PulpSolverError('{original}\nCBC exited non-zero solving "{name}". Its output was:\n{tail}'.format(
                    original=error,
                    name=lp.name,
                    tail=_solver_log_tail(log_path),
                )) from error
            finally:
                self.optionsDict.pop('logPath', None)
                try:
                    remove(log_path)
                except OSError:
                    pass

    return Diagnosable_PULP_CBC_CMD


def ilp_solver(timeout: float = ILP_SOLVER_TIMEOUT):
    '''The solver every ILP in this package should be handed.

    This exists because the timeout was previously requested as
    ``problem.solve(timeout=...)`` / ``problem.sequentialSolve(..., timeout=...)``.
    pulp has no such keyword: ``solve`` forwards **kwargs to
    ``PULP_CBC_CMD.actualSolve``, which does not take one, and ``sequentialSolve``
    does not accept **kwargs at all. Both therefore raised TypeError before the
    solver ever ran, so those code paths could only ever fail -- and, because the
    call sites catch the failure to dump a debug .lp, they failed noisily. The
    solver's real knob is ``timeLimit`` on the solver object, which is what this
    returns.
    '''
    return _diagnosable_cbc_class()(msg=SOLVER_MSG, timeLimit=timeout)


def failed_ilp_debug_path(file_name: str, force: bool = False):
    '''Absolute path for a failed-ILP artefact, or None if dumping is disabled.

    ``force=True`` is for the call sites that already sit behind an explicit
    ``debug=`` stream passed in by the caller: that is opt-in enough on its own,
    so it does not additionally need the env var. It still redirects the file out
    of the working directory, which is the point.

    Creates the directory on first use so callers do not each have to.
    '''
    if not (WRITE_FAILED_ILP_DEBUG or force):
        return None
    from os import makedirs
    makedirs(FAILED_ILP_DEBUG_DIR, exist_ok=True)
    return join(FAILED_ILP_DEBUG_DIR, file_name)


configure_default_solver()
