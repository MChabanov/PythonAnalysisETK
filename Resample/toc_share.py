#!/usr/bin/env python3
"""Share parsed HDF5 tables of contents between the variables of a 3D run.

Why
---
postcactus learns which datasets - (iteration, refinement level, component) -
a Carpet HDF5 file holds by listing every dataset name in it: the file's table
of contents (TOC), ``GridH5File._parse_toc``. On Lustre that costs ~0.3 s per
3D output file when cold, and a 3D variable has one file per MPI process per
output directory (64 on the 30M BBH run). The SimDir pickle keeps the TOC of
only the one file per directory that the iteration query looks at, so every
slice of every variable re-listed the other 63: 14-34 s of a 40-70 s cold
slice.

But the files written by one process into one output directory hold the same
datasets whatever the variable: ``alp.xyz.file_12.h5`` and
``gxx.xyz.file_12.h5`` both have exactly the components process 12 owned at
each output iteration. So a TOC parsed for one variable serves all of them.
:class:`TocShare` keeps parsed TOCs keyed by ``(directory, file suffix)``,
hands them to files that have not parsed their own, and can persist them
between runs.

When a TOC is shared
--------------------
Only when the variables were demonstrably output together. Per variable and
output directory, before anything is shared, the one file the iteration query
parsed for that variable must have a TOC *identical* to the shared TOC of the
same file - same iterations, levels and components, in the same order. The
first variable to reach a directory provides its TOCs; a variable that fails
the check (written at other iterations, say) keeps parsing its own files and
never contributes. Each shared TOC must in addition list the same iterations
as that reference file, which rejects a persisted TOC from before a directory
was complete. And should a shared TOC ever name a dataset the file lacks, the
3D backend redoes the slice with the variable's own TOCs
(:meth:`TocShare.reject`).

A shared TOC gives a file exactly the component lists its own parse would
have - same order too, since the dataset names differ only in the
variable-name prefix - so the output is bit-identical with or without sharing.

Persistence
-----------
With a cache file (``toc_cache`` in the config) the TOCs are loaded at the
start of a run, and those parsed during the run are gathered from all ranks
and merged into it at the end. Entries are plain dicts keyed by directory, a
few kB per file, so one cache can serve every 3D analysis of a simulation; a
later run then parses no TOC for the directories it already holds.
"""

import os
import pickle
import re

from postcactus.cactus_grid_h5 import GridSourceTOC

FORMAT = "postcactus-3d-toc-cache"
VERSION = 1

# The file-name part that does not depend on the variable (or group) name:
# "Bx.xyz.file_12.h5" -> ".xyz.file_12.h5", "rho_b.h5" -> ".h5".
_FILE_SUFFIX = re.compile(r"^.*?((?:\.xyz)?(?:\.file_\d+)?\.h5)$")


def file_key(h5file):
    """Variable-independent identity of a 3D output file: ``(directory, suffix)``.

    ``.../HDF5_3D/Bx.xyz.file_12.h5`` -> ``(".../HDF5_3D", ".xyz.file_12.h5")``.
    A reader without a path gets its ``id()``: it is never shared.
    """
    path = getattr(h5file, "_path", None)
    if path is None:
        return id(h5file)
    head, base = os.path.split(path)
    match = _FILE_SUFFIX.match(base)
    return (head, match.group(1) if match else base)


def _toc_from_frames(frames):
    """A finalized postcactus TOC from ``{it: {level: [components]}}``."""
    toc = GridSourceTOC()
    toc._frames = {int(it): {int(level): list(comps)
                             for level, comps in levels.items()}
                   for it, levels in frames.items()}
    toc.finalize()
    return toc


def _iteration_sets_agree(entries):
    """True if every TOC in ``{suffix: frames}`` lists the same iterations."""
    sets = {frozenset(frames) for frames in entries.values()}
    return len(sets) <= 1


class TocShare(object):
    """Parsed TOCs of one process, shared across the variables it reads.

    Usage per slice (one variable, one output directory's ``files``):
    :meth:`prepare` before the files are touched, :meth:`contribute` after.
    Counters: ``shared`` files given a TOC, ``loaded`` TOCs read from a cache.
    """

    def __init__(self):
        self._tocs = {}     # {directory: {suffix: GridSourceTOC}}
        self._compat = {}   # {(field, directory): may share}
        self._parsed = {}   # {directory: {suffix: frames}}, parsed in this process
        self.shared = 0
        self.loaded = 0

    # -- during a run -------------------------------------------------------

    def prepare(self, field, files):
        """Give unparsed ``files`` a shared TOC where that is safe.

        ``files`` are one field's files in one output directory, as
        ``GridReader._get_files`` returns them: ``files[0]`` is the file the
        iteration query parsed. Returns how many files got a shared TOC.
        """
        directory = self._directory(files)
        if directory is None or not self._compatible(field, directory, files):
            return 0
        shared = self._tocs.get(directory)
        if not shared:
            return 0      # the first variable here: it parses, then contributes
        ref = files[0]
        ref_iters = ref.get_toc().get_iters()
        n = 0
        for h5file in files:
            if h5file._toc is not None:
                continue
            toc = shared.get(file_key(h5file)[1])
            if toc is None or toc.get_iters() != ref_iters:
                continue
            # Exactly what _parse_toc would set, bar the field list, which
            # only the iteration query uses (it parses on demand if asked).
            h5file._toc, h5file._thorn, h5file._lama = toc, ref._thorn, ref._lama
            h5file._toc_shared = True
            n += 1
        self.shared += n
        return n

    def contribute(self, field, files):
        """Offer the TOCs ``files`` parsed themselves to the other variables."""
        directory = self._directory(files)
        if directory is None or not self._compat.get((field, directory)):
            return
        shared = self._tocs.setdefault(directory, {})
        for h5file in files:
            if h5file._toc is None or getattr(h5file, "_toc_shared", False):
                continue
            suffix = file_key(h5file)[1]
            if suffix not in shared:
                shared[suffix] = h5file._toc
                self._parsed.setdefault(directory, {})[suffix] = \
                    h5file._toc._frames

    def reject(self, field, files):
        """A shared TOC named a dataset missing from ``files``: stop sharing.

        Takes the shared TOCs back (the files parse their own on next use) and
        excludes this field in this directory from sharing for the rest of the
        run. Returns False if none of ``files`` had a shared TOC - the missing
        dataset is then not this module's doing.
        """
        mine = [f for f in files if getattr(f, "_toc_shared", False)]
        if not mine:
            return False
        self._compat[(field, self._directory(files))] = False
        for h5file in mine:
            h5file._toc = h5file._thorn = None
            h5file._toc_shared = False
        return True

    @staticmethod
    def _directory(files):
        if not files:
            return None
        key = file_key(files[0])
        return key[0] if isinstance(key, tuple) else None

    def _compatible(self, field, directory, files):
        key = (field, directory)
        if key not in self._compat:
            self._compat[key] = self._check(directory, files)
        return self._compat[key]

    def _check(self, directory, files):
        """May ``files`` share the TOCs held for ``directory``?"""
        shared = self._tocs.get(directory)
        if not shared:
            return True   # the first variable here provides the TOCs
        # Compare on the file the query parsed if the shared set has it,
        # otherwise on the first file it does have (parsing that one).
        for h5file in files:
            theirs = shared.get(file_key(h5file)[1])
            if theirs is not None:
                return h5file.get_toc()._frames == theirs._frames
        return False      # no file in common: nothing to share, nothing to vouch for

    # -- persistence --------------------------------------------------------

    def parsed(self):
        """``{directory: {suffix: frames}}`` parsed (not loaded) in this process."""
        return self._parsed

    def load(self, path):
        """Add the TOCs of a cache file written by :func:`save`.

        A missing file is an empty cache. Returns the number of TOCs loaded.
        """
        blob = _read(path)
        n = 0
        for directory, entries in blob["tocs"].items():
            shared = self._tocs.setdefault(directory, {})
            for suffix, frames in entries.items():
                if suffix not in shared:
                    shared[suffix] = _toc_from_frames(frames)
                    n += 1
        self.loaded += n
        return n


def _read(path):
    try:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
    except FileNotFoundError:
        return {"format": FORMAT, "version": VERSION, "tocs": {}}
    if not isinstance(blob, dict) or blob.get("format") != FORMAT \
            or blob.get("version") != VERSION:
        raise ValueError("%s is not a TOC cache (format %s v%d)"
                         % (path, FORMAT, VERSION))
    return blob


def save(path, parts):
    """Merge ``parts`` (each a :meth:`TocShare.parsed` result) into ``path``.

    Re-reads the file first, so TOCs another run added meanwhile are kept, and
    replaces it atomically. A directory whose new TOCs list other iterations
    than the stored ones (a simulation that was still writing to it) is
    replaced as a whole rather than mixed. Returns ``(directories, files)``
    written in this call.
    """
    new = {}
    for part in parts:
        for directory, entries in part.items():
            new.setdefault(directory, {}).update(entries)
    blob = _read(path)
    stored = blob["tocs"]
    n_files = 0
    for directory, entries in new.items():
        merged = dict(stored.get(directory, {}))
        merged.update(entries)
        stored[directory] = merged if _iteration_sets_agree(merged) \
            else dict(entries)
        n_files += len(entries)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return len(new), n_files
