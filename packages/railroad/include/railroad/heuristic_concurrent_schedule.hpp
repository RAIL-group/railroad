#pragma once

// Core of the concurrent heuristic: list-scheduling goals onto agents.
//
// Each goal fact not yet true is one unit (with the `found X` an `at X L`
// implies). In priority order, each goes to the agent that would finish it
// earliest from that agent's ready time and position. Longest-first (LPT)
// suits the makespan and shortest-first (SPT) the sum of completion times;
// both are tried and the one the value prefers is kept.
//
// Jobs: the units are cut from the team's relaxed plan for all the goals at
// once. Actions linked through a fluent that names an agent and
// something else (`holding r X`) are one agent's work, and so is establishing
// what that agent then handles (finding X); moves, and links through an
// agent's own state (`hand-full r`), are left out, since whoever does the job
// sees to those. Every other link between two jobs, through a fluent that
// names no agent (`at pot l`, a door opened), makes the consumer wait for the
// provider: its relaxed plan stops at that subgoal, and its route waits until
// the provider is done. Goal facts in one job are one unit; work no goal fact
// needs directly becomes a provider job. A job is scheduled once its
// providers are, so other agents' work shortens it, and a provider goes to the
// agent that lets the jobs waiting on it finish soonest (HEFT's lookahead), so
// it does not take the agent they need. With nothing to split, each goal fact
// is its own job and this is the list schedule above.
//
// Object-search refinement: a provider that only searches for an object Y is a
// search job, costed as an expected search, and leaves Y wherever it turns up
// (a distribution over places); an object a goal job delivers somewhere is
// provided there. An object another job uses where it is now is either used
// where it is delivered (the user waits for the delivery) or used where it is
// and then delivered (the delivery waits for the user); both are scheduled and
// the better kept (heuristic_concurrent.hpp).
//
// Each scheduled unit also keeps the distribution of its completion time over
// where its search succeeds, for valuing hedges (heuristic_concurrent.hpp).

#include "railroad/heuristic_concurrent_plans.hpp"

#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_set>

namespace railroad {
namespace concurrent {

// A completion time's distribution: (time, probability) pairs summing to 1.
using Dist = std::vector<std::pair<double, double>>;

namespace detail {
inline double survival(const Dist &d, double t) {  // P(T > t)
  double s = 0.0;
  for (const auto &[x, p] : d) {
    if (x > t) s += p;
  }
  return s;
}
// E[min over sets of the max over each set] for independent completion times.
inline double expected_min_of_max(const std::vector<std::vector<const Dist *>> &sets) {
  std::vector<double> ts{0.0};
  for (const auto &set : sets) {
    for (const Dist *d : set) {
      for (const auto &[x, p] : *d) ts.push_back(std::max(0.0, x));
    }
  }
  std::sort(ts.begin(), ts.end());
  ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
  double e = 0.0;
  for (std::size_t k = 1; k < ts.size(); ++k) {
    double t = ts[k - 1], s_min = 1.0;  // P(min over sets of max > t)
    for (const auto &set : sets) {
      double f_max = 1.0;  // P(max <= t)
      for (const Dist *d : set) f_max *= 1.0 - survival(*d, t);
      s_min *= 1.0 - f_max;
    }
    e += (ts[k] - t) * s_min;
  }
  return e;
}
}  // namespace detail

// E[max] of independent completion times.
inline double expected_max(const std::vector<const Dist *> &set) {
  return detail::expected_min_of_max({set});
}
// E[min(max A, max B)] of independent completion times: the time until either
// of two options is done.
inline double expected_min_of_max(const std::vector<const Dist *> &a,
                                  const std::vector<const Dist *> &b) {
  return detail::expected_min_of_max({a, b});
}

// A unit of work for one agent.
struct Job {
  std::vector<int> roots;      // goal facts, or what a provider job provides
  bool goal = true;            // do its roots count toward the value?
  int companion = -1;          // the `found X` of the object it carries
  int search = -1;             // a search job's `found Y`
  std::vector<int> provided;   // subgoals other jobs provide (item_index)
  std::vector<int> providers;  // the job providing each
  double finish_fixed = -1.0;  // >= 0: achieved by an effect in flight
  std::vector<GoalPlan> plans;  // per agent
  double key = -1.0;
};

// One goal branch's jobs and their plans, scheduled on demand (all agents, or
// a subset of them for a hedge).
class BranchSchedule {
 public:
  struct Run {
    double makespan = 0.0, sum = 0.0;
    std::vector<double> free_at;  // per agent, when it is done with this run
    std::vector<int> end_loc;
    std::vector<int> done_goal;   // per completed goal fact
    std::vector<double> done_at;
    std::vector<int> done_agent;  // -1: in flight, or the team relaxation
    std::vector<Dist> done_dist;
    std::vector<std::pair<int, int>> searches;  // (search job's found Y, agent)
    bool complete = true;  // false: some job had no agent to do it
    double value(const ConcurrentHeuristicOptions &o) const {
      return o.lambda_add * sum + o.lambda_ms * makespan;
    }
    std::vector<const Dist *> dists() const {
      std::vector<const Dist *> out;
      for (const auto &d : done_dist) out.push_back(&d);
      return out;
    }
  };

  // passes[0] is the unrestricted pass; passes[r + 1] agent r's, when the
  // schedule is per agent.
  // `in_place`: an object another job uses where it is now is delivered
  // after that use, not before it (see the top).
  void build(Context &cx, std::vector<Pass> &passes, const std::vector<int> &goals,
             bool in_place = false) {
    cx_ = &cx;
    in_place_ = in_place;
    threatened_ = false;
    passes_ = &passes;
    const Problem &pb = cx.pb;
    const ConcurrentHeuristicOptions &opts = cx.opts;
    Pass &U = passes[0];
    per_agent_ = opts.agent_aware && pb.num_agents() > 1;
    n_agents_ = per_agent_ ? pb.num_agents() : 1;
    // With a single agent its route can still be chained.
    solo_ = (!per_agent_ && pb.num_agents() == 1) ? 0 : -1;

    ready_.assign(n_agents_, 0.0);
    start_loc_.assign(n_agents_, -1);
    for (std::size_t r = 0; r < n_agents_; ++r) {
      Pass &P = pass_of(r);
      ready_[r] = per_agent_ ? P.cost[pb.agent_free[r]] : team_ready(pb, U);
      if (agent_of(r) >= 0) start_loc_[r] = agent_location(pb, P, agent_of(r));
    }

    // `at X L` and the `found X` it implies are one job: whoever brings X to
    // L must find it first. Planned apart, `at X L` could be "achieved" by
    // searching L in the hope X is there while another agent finds X.
    auto companion_of = [&](int g) {
      if (!opts.joint_found) return -1;
      int fx = pb.found_of[g];
      if (fx < 0 || std::find(goals.begin(), goals.end(), fx) == goals.end()) return -1;
      if (U.best[fx] == Pass::AVAIL && U.cost[fx] <= 1e-9) return -1;
      return fx;
    };
    std::vector<int> paired;
    for (int g : goals) {
      int fx = companion_of(g);
      if (fx >= 0) paired.push_back(fx);
    }
    jobs_.clear();
    for (int g : goals) {
      if (U.best[g] == Pass::AVAIL && U.cost[g] <= 1e-9) continue;
      if (std::find(paired.begin(), paired.end(), g) != paired.end()) continue;
      Job j;
      j.roots = {g};
      j.companion = companion_of(g);
      j.finish_fixed = (U.best[g] == Pass::AVAIL) ? U.cost[g] : -1.0;
      jobs_.push_back(std::move(j));
    }
    if (per_agent_ || solo_ >= 0) form_jobs(cx, U);

    for (Job &j : jobs_) {
      if (j.finish_fixed >= 0.0 || j.search >= 0) continue;
      j.plans.resize(n_agents_);
      for (std::size_t r = 0; r < n_agents_; ++r) {
        if (!std::isfinite(ready_[r])) continue;
        j.plans[r] = plan_goal(cx, pass_of(r), j.roots, j.companion, agent_of(r), ready_[r],
                               j.provided);
      }
    }
    compute_keys();
  }

  // The better of the longest- and shortest-first schedules. `after`: start
  // each agent when and where another run leaves it, except those in `skip`.
  Run best_run(const Run *after = nullptr, const std::vector<char> &skip = {}) {
    const ConcurrentHeuristicOptions &opts = cx_->opts;
    std::vector<std::size_t> lpt(jobs_.size());
    std::iota(lpt.begin(), lpt.end(), 0);
    std::stable_sort(lpt.begin(), lpt.end(),
                     [this](std::size_t a, std::size_t b) { return jobs_[a].key > jobs_[b].key; });
    Run best = run(lpt, after, skip);
    if (jobs_.size() > 1) {
      std::vector<std::size_t> spt = lpt;
      std::stable_sort(spt.begin(), spt.end(),
                       [this](std::size_t a, std::size_t b) { return jobs_[a].key < jobs_[b].key; });
      if (spt != lpt) {
        Run alt = run(spt, after, skip);
        if (alt.value(opts) < best.value(opts) - 1e-9) best = std::move(alt);
      }
    }
    return best;
  }

  bool per_agent() const { return per_agent_; }
  // Does a job use an object where it is now that another job delivers?
  bool threatened() const { return threatened_; }

 private:
  Context *cx_ = nullptr;
  std::vector<Pass> *passes_ = nullptr;
  bool per_agent_ = false;
  std::size_t n_agents_ = 1;
  int solo_ = -1;
  bool in_place_ = false, threatened_ = false;
  std::vector<double> ready_;
  std::vector<int> start_loc_;
  std::vector<Job> jobs_;
  std::vector<uint32_t> cover_;
  uint32_t cover_gen_ = 0;

  Pass &pass_of(std::size_t r) { return per_agent_ ? (*passes_)[r + 1] : (*passes_)[0]; }
  int agent_of(std::size_t r) const { return per_agent_ ? static_cast<int>(r) : solo_; }

  // Serial "team" agent: ready when the first agent is.
  static double team_ready(const Problem &pb, const Pass &U) {
    double r = INF;
    for (int f : pb.agent_free) r = std::min(r, U.cost[f]);
    return std::isfinite(r) ? r : 0.0;
  }

  // Cut the team's relaxed plan for every open goal into jobs (see the top).
  void form_jobs(Context &cx, Pass &U) {
    const Problem &pb = cx.pb;
    const ConcurrentHeuristicOptions &opts = cx.opts;
    std::vector<int> roots, companions;
    for (const Job &j : jobs_) {
      if (j.finish_fixed >= 0.0) continue;
      roots.push_back(j.roots.front());
      if (j.companion >= 0) companions.push_back(j.companion);
    }
    if (roots.empty()) return;
    // Extracted as plan_goal would, forcing included.
    std::vector<std::pair<int, int>> forced;
    if (opts.expected_search && opts.route_chaining) {
      for (int g : roots) {
        if (!U.uncertain(pb, g)) continue;
        int d = deterministic_achiever(pb, U, g);
        if (d >= 0) {
          forced.push_back({g, U.best[g]});
          U.best[g] = d;
        }
      }
    }
    struct Restore {
      Pass &P;
      std::vector<std::pair<int, int>> &forced;
      ~Restore() {
        for (const auto &[g, from] : forced) P.best[g] = from;
      }
    } restore{U, forced};
    cx.ex.next_stamp(pb);
    Extraction ex;
    if (!forced.empty()) {
      for (int g : roots) cx.ex.extract(pb, cx.ts, U, {g}, ex);
      for (int c : companions) cx.ex.extract(pb, cx.ts, U, {c}, ex);
    } else {
      std::vector<int> all(roots);
      all.insert(all.end(), companions.begin(), companions.end());
      cx.ex.extract(pb, cx.ts, U, all, ex);
    }

    // The plan's non-move actions, and the links between them.
    std::vector<int> acts;
    std::unordered_map<int, int> slot;
    for (int a : ex.actions) {
      int ag = pb.acts[a].agent;
      if (ag >= 0 && pb.move_dest(a, ag) >= 0) continue;
      slot[a] = static_cast<int>(acts.size());
      acts.push_back(a);
    }
    struct Link { int from, to, fluent; };
    std::vector<Link> links;
    for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
      for (int q : pb.acts[acts[i]].pre) {
        int b = U.best[q];
        if (b < 0) continue;
        auto it = slot.find(b);
        if (it == slot.end() || it->second == i) continue;
        links.push_back({it->second, i, q});
      }
    }
    // Links through an agent's own state (`hand-full r`, where it is) tie
    // nothing: whoever does a job sees to that itself.
    auto bound = [&](int q) { return pb.loc_agent[q] < 0 && pb.names_agent_and_more(q); };
    auto own_state = [&](int q) { return pb.loc_agent[q] >= 0 || pb.names_agent(q); };
    links.erase(std::remove_if(links.begin(), links.end(),
                               [&](const Link &l) { return own_state(l.fluent) && !bound(l.fluent); }),
                links.end());
    std::vector<int> parent(acts.size());
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
    for (const Link &l : links) {
      if (bound(l.fluent)) parent[find(l.from)] = find(l.to);
    }
    // So are actions needing the same such fluent, even one already true
    // (boiling an egg and putting it in a bowl both need it held).
    std::unordered_map<int, int> first_user;
    for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
      for (int q : pb.acts[acts[i]].pre) {
        if (!bound(q)) continue;
        auto [it, fresh] = first_user.emplace(q, i);
        if (!fresh) parent[find(i)] = find(it->second);
      }
    }
    // What each job handles: the non-agent arguments of its agent-bound
    // fluents. A job also establishes facts about what it handles.
    auto is_agent = [&](const std::string &s) {
      return std::find(pb.agent_names.begin(), pb.agent_names.end(), s) != pb.agent_names.end();
    };
    std::vector<std::unordered_set<std::string>> held(acts.size());
    auto collect = [&]() {
      for (auto &h : held) h.clear();
      for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
        const CompiledAction &ca = pb.acts[acts[i]];
        auto take = [&](int f) {
          if (!bound(f)) return;
          for (const auto &a : pb.args(f)) {
            if (!is_agent(a)) held[find(i)].insert(a);
          }
        };
        for (int q : ca.pre) take(q);
        for (const auto &ad : ca.adds) take(ad.fluent);
      }
    };
    for (bool changed = true; changed;) {
      changed = false;
      collect();
      for (const Link &l : links) {
        int a = find(l.to), b = find(l.from);
        if (a == b || bound(l.fluent)) continue;
        for (const auto &x : pb.args(l.fluent)) {
          if (held[a].count(x)) {
            parent[b] = a;
            changed = true;
            break;
          }
        }
        if (changed) break;
      }
    }
    collect();

    // Goal facts join the job of the action supporting them; one job per
    // part of the plan.
    std::unordered_map<int, int> job_of_part;  // part (root slot) -> new job
    std::vector<Job> out;
    std::vector<int> part_of_job;
    for (Job &j : jobs_) {
      if (j.finish_fixed >= 0.0) {
        out.push_back(std::move(j));
        part_of_job.push_back(-1);
        continue;
      }
      int s = U.best[j.roots.front()];
      auto it = slot.find(s);
      int part = it == slot.end() ? -1 : find(it->second);
      auto jt = part < 0 ? job_of_part.end() : job_of_part.find(part);
      if (jt == job_of_part.end()) {
        if (part >= 0) job_of_part[part] = static_cast<int>(out.size());
        out.push_back(std::move(j));
        part_of_job.push_back(part);
      } else {
        Job &m = out[jt->second];
        m.roots.push_back(j.roots.front());
        if (m.companion < 0) m.companion = j.companion;
      }
    }
    // A goal job with no `at` companion is anchored on what it carries, when
    // that is one object not yet found.
    for (std::size_t i = 0; i < out.size(); ++i) {
      Job &m = out[i];
      if (m.companion >= 0 || m.finish_fixed >= 0.0 || part_of_job[i] < 0) continue;
      int only = -1, count = 0;
      for (const auto &x : held[part_of_job[i]]) {
        int fx = pb.lookup(Fluent("found", {x}));
        if (fx < 0 || (U.best[fx] == Pass::AVAIL && U.cost[fx] <= 1e-9)) continue;
        only = fx;
        ++count;
      }
      if (count == 1 && U.reachable(only)) m.companion = only;
    }
    // An object a goal job delivers somewhere is used there, whatever the team
    // plan did with it on the way.
    std::unordered_map<int, int> deliverer;  // object -> job
    for (std::size_t i = 0; i < out.size(); ++i) {
      for (int g : out[i].roots) {
        if (pb.obj_of[g] >= 0 && pb.obj_place[g] >= 0) deliverer[pb.obj_of[g]] = static_cast<int>(i);
      }
    }
    auto delivered_by = [&](const Link &l) {
      auto it = deliverer.find(pb.obj_of[l.fluent]);
      return it == deliverer.end() ? -1 : it->second;
    };
    // Parts no goal needs directly, that some job waits on, are provider jobs.
    std::vector<char> wanted(acts.size(), 0);
    for (int part : part_of_job) {
      if (part >= 0) wanted[part] = 1;
    }
    for (bool grew = true; grew;) {
      grew = false;
      for (const Link &l : links) {
        int a = find(l.to), b = find(l.from);
        if (a != b && wanted[a] && !wanted[b] && delivered_by(l) < 0) wanted[b] = grew = 1;
      }
    }
    for (int p = 0; p < static_cast<int>(acts.size()); ++p) {
      if (find(p) != p || !wanted[p] || job_of_part.count(p)) continue;
      Job j;
      j.goal = false;
      // A search job: every action finds the same object.
      int y = -2;
      for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
        if (find(i) != p) continue;
        int found = -1;
        for (const auto &ad : pb.acts[acts[i]].adds) {
          if (ad.uncertain() && pb.is_found[ad.fluent]) found = ad.fluent;
        }
        y = (y == -2 || y == found) ? found : -1;
      }
      if (y >= 0 && !(U.best[y] == Pass::AVAIL && U.cost[y] <= 1e-9) && U.reachable(y)) {
        j.search = y;
      } else {
        for (const Link &l : links) {
          if (find(l.from) == p && find(l.to) != p &&
              std::find(j.roots.begin(), j.roots.end(), l.fluent) == j.roots.end()) {
            j.roots.push_back(l.fluent);
          }
        }
      }
      job_of_part[p] = static_cast<int>(out.size());
      out.push_back(std::move(j));
      part_of_job.push_back(p);
    }
    jobs_ = std::move(out);

    // Is the link through fluent q, an object's place, which job d delivers
    // elsewhere, a use of it in place? Then in place it is, if so built.
    auto in_place = [&](int q, int d) {
      if (pb.obj_place[q] < 0) return false;
      const std::vector<int> &roots = jobs_[d].roots;
      if (std::find(roots.begin(), roots.end(), q) != roots.end()) return false;
      threatened_ = true;
      return in_place_;
    };
    auto wait_for = [&](Job &j, int item, int provider) {
      for (std::size_t k = 0; k < j.provided.size(); ++k) {
        if (j.provided[k] == item && j.providers[k] == provider) return;
      }
      j.provided.push_back(item);
      j.providers.push_back(provider);
    };
    // What each job waits on, and from whom: an object a job delivers or a
    // search job finds, else the linked fluent itself.
    for (const Link &l : links) {
      int a = find(l.to), b = find(l.from);
      if (a == b) continue;
      auto ja = job_of_part.find(a);
      if (ja == job_of_part.end()) continue;
      const int obj = pb.obj_of[l.fluent];
      int provider = delivered_by(l), item = obj >= 0 ? object_item(obj) : l.fluent;
      if (provider >= 0 && provider != ja->second && in_place(l.fluent, provider)) {
        // Used where it is found, before it is delivered (see below).
        wait_for(jobs_[provider], l.fluent, ja->second);
        continue;
      }
      if (provider < 0) {
        auto jb = job_of_part.find(b);
        if (jb == job_of_part.end()) continue;
        provider = jb->second;
        int fy = jobs_[provider].search;
        if (obj < 0 || fy < 0 || pb.obj_of[fy] != obj) item = l.fluent;
      }
      if (provider == ja->second) continue;
      wait_for(jobs_[ja->second], item, provider);
    }
    // An object used where it is now, which another job delivers elsewhere:
    // the team plan has no link to order the two. Used where it is delivered,
    // the user waits for the delivery; used in place, the delivery waits there
    // for the user.
    for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
      auto ja = job_of_part.find(find(i));
      if (ja == job_of_part.end()) continue;
      const CompiledAction &ca = pb.acts[acts[i]];
      for (int q : ca.pre) {
        int y = pb.obj_of[q];
        if (y < 0 || pb.obj_place[q] < 0) continue;
        if (std::find(ca.consumes.begin(), ca.consumes.end(), q) != ca.consumes.end()) continue;
        auto d = deliverer.find(y);
        if (d == deliverer.end() || d->second == ja->second) continue;
        threatened_ = true;
        if (in_place_) {
          wait_for(jobs_[d->second], q, ja->second);
        } else {
          wait_for(jobs_[ja->second], object_item(y), d->second);
        }
      }
    }
    break_cycles();
  }

  // Providers forming a cycle are dropped: those jobs plan the subgoals
  // themselves.
  void break_cycles() {
    const std::size_t n = jobs_.size();
    std::vector<int> state(n, 0);  // 0 new, 1 on stack, 2 done
    std::function<bool(std::size_t)> cyclic = [&](std::size_t i) {
      if (state[i] == 1) return true;
      if (state[i] == 2) return false;
      state[i] = 1;
      bool c = false;
      for (int p : jobs_[i].providers) c = c || cyclic(static_cast<std::size_t>(p));
      state[i] = 2;
      return c;
    };
    for (std::size_t i = 0; i < n; ++i) {
      std::fill(state.begin(), state.end(), 0);
      if (!cyclic(i)) continue;
      jobs_[i].provided.clear();
      jobs_[i].providers.clear();
    }
  }

  // A search job on agent r from `from` at time t: the expected time to find
  // the object, and where.
  double search_load(std::size_t r, int from, double t, int fy, std::vector<Found> &where) {
    Context &cx = *cx_;
    auto rest = [](int, bool, double) { return 0.0; };
    return expected_search(cx.pb, cx.ts, pass_of(r), fy, -1, agent_of(r), from, t, rest, where);
  }

  static Provision provision_from(const Problem &pb, const std::vector<Found> &where) {
    Provision pv;
    double found = 0.0;
    for (const auto &w : where) found += w.prob;
    if (found <= 0.0) return pv;
    for (const auto &w : where) {
      int place = w.loc >= 0 ? pb.loc_place[w.loc] : -1;
      if (place < 0) continue;
      pv.place.push_back(place);
      pv.prob.push_back(w.prob / found);
      pv.time.push_back(w.t);
    }
    return pv;
  }

  // Where and when provided[k] of job j holds, given its provider's search
  // outcome or finish.
  Provision provision_for(const Job &j, std::size_t k, const Provision &searched,
                          double finish) const {
    const Problem &pb = cx_->pb;
    const Job &p = jobs_[j.providers[k]];
    int item = j.provided[k];
    if (p.search >= 0) return searched;
    int obj = item_object(item);
    if (obj >= 0) {
      for (int g : p.roots) {
        if (pb.obj_of[g] == obj && pb.obj_place[g] >= 0) return Provision{{pb.obj_place[g]}, {1.0}, {finish}};
      }
    } else if (pb.obj_place[item] >= 0) {
      return Provision{{pb.obj_place[item]}, {1.0}, {finish}};
    }
    return Provision{{}, {}, {finish}};
  }

  // Priorities: each job's duration as the first job of the agent that would
  // finish it soonest, with its providers assumed scheduled that way.
  void compute_keys() {
    const Problem &pb = cx_->pb;
    std::vector<Provision> searched(jobs_.size());
    std::vector<double> finish(jobs_.size(), INF);
    std::vector<char> done(jobs_.size(), 0);
    for (std::size_t round = 0; round <= jobs_.size(); ++round) {
      for (std::size_t i = 0; i < jobs_.size(); ++i) {
        if (done[i]) continue;
        Job &j = jobs_[i];
        bool ready = true;
        for (int p : j.providers) ready = ready && done[p];
        if (!ready) continue;
        done[i] = 1;
        if (j.finish_fixed >= 0.0) {
          j.key = -1.0;
          finish[i] = j.finish_fixed;
          continue;
        }
        if (j.search >= 0) {
          double best = INF;
          for (std::size_t r = 0; r < n_agents_; ++r) {
            if (!std::isfinite(ready_[r])) continue;
            std::vector<Found> where;
            double e = search_load(r, start_loc_[r], ready_[r], j.search, where);
            if (e < best) {
              best = e;
              searched[i] = provision_from(pb, where);
              finish[i] = ready_[r] + e;
            }
          }
          j.key = best;
          continue;
        }
        std::vector<Provision> pv(j.provided.size());
        std::vector<const Provision *> ptr(j.provided.size(), nullptr);
        for (std::size_t k = 0; k < j.provided.size(); ++k) {
          int p = j.providers[k];
          if (!std::isfinite(finish[p])) continue;
          pv[k] = provision_for(j, k, searched[p], finish[p]);
          ptr[k] = &pv[k];
        }
        double m = INF;
        for (std::size_t r = 0; r < n_agents_; ++r) {
          if (!std::isfinite(ready_[r]) || !j.plans[r].ok) continue;
          const GoalPlan &tp = j.plans[r];
          Duration d = goal_duration(*cx_, pass_of(r), agent_of(r), start_loc_[r], ready_[r], tp, ptr);
          double load = std::max(d.serial, bound(tp)) + d.delta;
          if (load < m) {
            m = load;
            finish[i] = ready_[r] + load;
          }
        }
        j.key = m;
      }
    }
  }

  static Dist dist_of(const Duration &d, double start, double load) {
    if (d.where.empty()) return Dist{{start + load, 1.0}};
    // Shifted so its mean is the scheduled finish (bounds and deltas included).
    double mean = d.residual * d.residual_done;
    for (const auto &w : d.where) mean += w.prob * w.t;
    double shift = start + load - mean;
    Dist out;
    for (const auto &w : d.where) out.push_back({w.t + shift, w.prob});
    if (d.residual > 1e-12) out.push_back({d.residual_done + shift, d.residual});
    return out;
  }

  // A schedule being built: when and where each agent is free, and what each
  // scheduled job left behind.
  struct Board {
    std::vector<double> free_at;
    std::vector<int> end_loc;
    std::vector<int> n_assigned;
    std::vector<char> scheduled;
    std::vector<double> finish;
    std::vector<Provision> searched;
  };
  // A job on one agent.
  struct Placement {
    double finish = INF, load = 0.0;
    Duration d;                // a goal or provider job's
    std::vector<Found> where;  // a search job's outcomes
  };

  // Where and when what job j waits on holds, as far as the board has it.
  void provisions(const Job &j, const Board &b, std::vector<Provision> &pv,
                  std::vector<const Provision *> &ptr) const {
    pv.assign(j.provided.size(), Provision{});
    ptr.assign(j.provided.size(), nullptr);
    for (std::size_t k = 0; k < j.provided.size(); ++k) {
      int p = j.providers[k];
      if (!std::isfinite(b.finish[p])) continue;
      pv[k] = provision_for(j, k, b.searched[p], b.finish[p]);
      ptr[k] = &pv[k];
    }
  }

  Placement place(std::size_t ji, std::size_t r, const Board &b,
                  const std::vector<const Provision *> &ptr) {
    const Job &j = jobs_[ji];
    Placement p;
    if (!std::isfinite(b.free_at[r])) return p;
    if (j.search >= 0) {
      p.load = search_load(r, b.end_loc[r], b.free_at[r], j.search, p.where);
    } else {
      const GoalPlan &tp = j.plans[r];
      if (!tp.ok) return p;
      p.d = goal_duration(*cx_, pass_of(r), agent_of(r), b.end_loc[r], b.free_at[r], tp, ptr);
      // The relaxed critical path only bounds an agent's first goal.
      p.load = (b.n_assigned[r] == 0 ? std::max(p.d.serial, bound(tp)) : p.d.serial) + p.d.delta;
    }
    p.finish = b.free_at[r] + p.load;
    return p;
  }

  void commit(std::size_t ji, std::size_t r, const Placement &p, Board &b) const {
    const Problem &pb = cx_->pb;
    b.free_at[r] = b.finish[ji] = p.finish;
    b.n_assigned[r] += 1;
    b.scheduled[ji] = 1;
    if (jobs_[ji].search >= 0) {
      b.searched[ji] = provision_from(pb, p.where);
      if (!b.searched[ji].place.empty()) {
        b.end_loc[r] = pb.place_loc_of(agent_of(r), b.searched[ji].mode());
      }
    } else if (p.d.end >= 0) {
      b.end_loc[r] = p.d.end;
    }
  }

  // The soonest job k could finish, on any agent not skipped.
  double soonest(std::size_t k, const Board &b, const std::vector<char> &skip) {
    if (jobs_[k].finish_fixed >= 0.0) return jobs_[k].finish_fixed;
    std::vector<Provision> pv;
    std::vector<const Provision *> ptr;
    provisions(jobs_[k], b, pv, ptr);
    double f = INF;
    for (std::size_t r = 0; r < n_agents_; ++r) {
      if (skip.empty() || !skip[r]) f = std::min(f, place(k, r, b, ptr).finish);
    }
    return f;
  }

  // The agent for job ji, or -1: the one that finishes it soonest, unless
  // other jobs can start once it is done. Then it is the one that lets those
  // finish soonest, so that a provider does not take the agent its consumer
  // needs (a one-step lookahead).
  int choose(std::size_t ji, const Board &b, const std::vector<char> &skip, Placement &best) {
    std::vector<Provision> pv;
    std::vector<const Provision *> ptr;
    provisions(jobs_[ji], b, pv, ptr);
    std::vector<std::size_t> waiting;
    for (std::size_t k = 0; k < jobs_.size(); ++k) {
      const std::vector<int> &ps = jobs_[k].providers;
      if (b.scheduled[k] || std::find(ps.begin(), ps.end(), static_cast<int>(ji)) == ps.end()) continue;
      bool ready = true;
      for (int p : ps) ready = ready && (b.scheduled[p] || p == static_cast<int>(ji));
      if (ready) waiting.push_back(k);
    }
    int best_r = -1;
    double best_score = INF;
    for (std::size_t r = 0; r < n_agents_; ++r) {
      if (!skip.empty() && skip[r]) continue;
      Placement p = place(ji, r, b, ptr);
      if (!std::isfinite(p.finish)) continue;
      double score = p.finish;
      if (!waiting.empty()) {
        Board then = b;
        commit(ji, r, p, then);
        for (std::size_t k : waiting) score = std::max(score, soonest(k, then, skip));
      }
      if (best_r < 0 || score < best_score || (score == best_score && p.finish < best.finish)) {
        best_r = static_cast<int>(r);
        best_score = score;
        best = std::move(p);
      }
    }
    return best_r;
  }

  // One list-scheduling pass in the given priority order. A job waits until
  // its providers are scheduled.
  Run run(const std::vector<std::size_t> &order, const Run *after, const std::vector<char> &skip) {
    Context &cx = *cx_;
    const Problem &pb = cx.pb;
    Pass &U = (*passes_)[0];
    Run out;
    Board b;
    b.free_at = after ? after->free_at : ready_;
    b.end_loc = after ? after->end_loc : start_loc_;
    b.n_assigned.assign(n_agents_, 0);
    b.scheduled.assign(jobs_.size(), 0);
    b.finish.assign(jobs_.size(), INF);
    b.searched.assign(jobs_.size(), Provision{});
    // Fluents achieved along an assigned goal's plan (e.g. `found X` on the
    // way to `at X L`) need no goal of their own.
    if (cover_.size() != pb.num_fluents()) cover_.assign(pb.num_fluents(), 0);
    if (++cover_gen_ == 0) {
      std::fill(cover_.begin(), cover_.end(), 0);
      cover_gen_ = 1;
    }
    auto done_goal = [&](const Job &j, double at, int agent, const Dist &dist) {
      if (!j.goal) return;
      for (int g : j.roots) {
        out.done_goal.push_back(g);
        out.done_at.push_back(at);
        out.done_agent.push_back(agent);
        out.done_dist.push_back(dist);
      }
    };
    std::vector<std::size_t> pending(order);
    while (!pending.empty()) {
      std::size_t pick = pending.size();
      for (std::size_t k = 0; k < pending.size() && pick == pending.size(); ++k) {
        bool ready = true;
        for (int p : jobs_[pending[k]].providers) ready = ready && b.scheduled[p];
        if (ready) pick = k;
      }
      if (pick == pending.size()) break;  // providers form no cycle (break_cycles)
      const std::size_t ji = pending[pick];
      pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(pick));
      b.scheduled[ji] = 1;
      const Job &j = jobs_[ji];
      if (j.finish_fixed >= 0.0) {
        b.finish[ji] = j.finish_fixed;
        done_goal(j, j.finish_fixed, -1, Dist{{j.finish_fixed, 1.0}});
        continue;
      }
      if (j.goal) {
        bool covered = true;
        for (int g : j.roots) covered = covered && cover_[g] == cover_gen_;
        if (covered) continue;  // achieved along another goal's plan
      }
      Placement p;
      const int r = choose(ji, b, skip, p);
      if (r < 0) {
        out.complete = false;
        if (j.search >= 0) continue;  // no one can search: its users go without
        // No single agent can do it: fall back to the team relaxation.
        cx.ex.next_stamp(pb);
        Extraction ex;
        cx.ex.extract(pb, cx.ts, U, j.roots, ex);
        double at = 0.0;
        for (int g : j.roots) at = std::max(at, U.cost[g]);
        at += ex.delta;
        b.finish[ji] = at;
        done_goal(j, at, -1, Dist{{at, 1.0}});
        continue;
      }
      const double start = b.free_at[r];
      commit(ji, static_cast<std::size_t>(r), p, b);
      if (j.search >= 0) {
        out.searches.push_back({j.search, r});
        continue;
      }
      for (int f : j.plans[r].covers) cover_[f] = cover_gen_;
      done_goal(j, p.finish, r, dist_of(p.d, start, p.load));
    }
    for (double at : out.done_at) {
      out.makespan = std::max(out.makespan, at);
      out.sum += at;
    }
    out.free_at = std::move(b.free_at);
    out.end_loc = std::move(b.end_loc);
    return out;
  }
};

}  // namespace concurrent
}  // namespace railroad
