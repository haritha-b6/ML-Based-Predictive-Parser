"""Exact reference recognizer (exhaustive backtracking with memoisation).
Gives ground truth for validity and for the longest viable prefix, even for
grammars that are not LL(1). Works because the grammars have no left recursion."""
import sys
sys.setrecursionlimit(20000)


class Oracle:
    def __init__(self, g):
        self.g = g

    def analyse(self, toks):
        """Return (accepted, k): k = length of longest viable prefix."""
        g, n = self.g, len(toks)
        seen, best = set(), [0]

        def rec(stack, pos):
            key = (stack, pos)
            if key in seen:
                return False
            seen.add(key)
            if pos > best[0]:
                best[0] = pos
            if not stack:
                return pos == n
            s, rest = stack[0], stack[1:]
            if g.is_nt(s):
                for pi in g.by_lhs[s]:
                    if rec(g.productions[pi][1] + rest, pos):
                        return True
                return False
            return pos < n and toks[pos] == s and rec(rest, pos + 1)

        ok = rec((g.start,), 0)
        return ok, best[0]
