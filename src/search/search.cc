// Copyright 2013-2016 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <cmath>
#include <csignal>
#include <unistd.h>
#include <string>
#include <fstream>

#include "src/search/search.h"
#include "src/transform/weighted.h"

using namespace cpputil;
using namespace std;
using namespace std::chrono;
using namespace x64asm;

namespace {

bool give_up_now = false;
void handler(int sig, siginfo_t* siginfo, void* context) {
  give_up_now = true;
}

} // namespace

namespace stoke {

Search::Search(Transform* transform) : transform_(transform) {
  set_seed(0);
  set_timeout_itr(0);
  set_timeout_sec(steady_clock::duration::zero());
  set_beta(1.0);
  set_progress_callback(nullptr, nullptr);
  set_statistics_callback(nullptr, nullptr);
  set_statistics_interval(100000);

  static bool once = false;
  if (!once) {
    once = true;

    struct sigaction term_act;
    memset(&term_act, '\0', sizeof(term_act));
    sigfillset(&term_act.sa_mask);
    term_act.sa_sigaction = handler;
    term_act.sa_flags = SA_ONSTACK;

    sigaction(SIGINT, &term_act, 0);
  }
}

void Search::run_parallel_tempering(const Cfg &target, CostFunction &fxn, Init init,
                                    vector<SearchState> &replicas, vector<TUnit> &aux_fxns,
                                    const vector<double> &betas)
{

  const size_t num_replicas = betas.size();
  assert(replicas.size() == num_replicas);

  // Initialize all replicas
  for (size_t i = 0; i < num_replicas; ++i)
  {
    configure(target, fxn, replicas[i], aux_fxns);
    assert(replicas[i].best_yet.is_sound());
    assert(replicas[i].best_correct.is_sound());
  }

  vector<Statistics> stats(num_replicas);
  const auto start = chrono::steady_clock::now();
  give_up_now = false;
  size_t iterations = 0;

  while (!give_up_now)
  {
    for (size_t i = 0; i < num_replicas; ++i)
    {
      SearchState &state = replicas[i];
      const double beta = betas[i];

      // Basic SA step
      TransformInfo ti = (*transform_)(state.current);
      if (!ti.success)
        continue;

      const auto p = prob_(gen_);
      const auto max = state.current_cost - (log(p) / beta);
      const auto new_res = fxn(state.current, max + 1);
      const auto is_correct = new_res.first;
      const auto new_cost = new_res.second;

      if (new_cost > max)
      {
        (*transform_).undo(state.current, ti);
        continue;
      }

      state.current_cost = new_cost;
      if (new_cost < state.best_yet_cost)
      {
        state.best_yet = state.current;
        state.best_yet_cost = new_cost;
      }

      if (is_correct && ((new_cost == 0) || (new_cost < state.best_correct_cost)))
      {
        state.success = true;
        state.best_correct = state.current;
        state.best_correct_cost = new_cost;
        new_best_correct_cb_({state}, new_best_correct_cb_arg_);
      }

      if (state.current_cost == 0)
      {
        give_up_now = true;
        break;
      }
    }

    // Attempt replica exchange between neighboring temperatures
    for (size_t i = 0; i < num_replicas - 1; ++i)
    {
      auto &s1 = replicas[i];
      auto &s2 = replicas[i + 1];

      double delta = (betas[i] - betas[i + 1]) * (s2.current_cost - s1.current_cost);
      double prob_swap = exp(min(0.0, delta));

      if (prob_(gen_) < prob_swap)
      {
        swap(s1.current, s2.current);
        swap(s1.current_cost, s2.current_cost);
      }
    }

    if (++iterations >= timeout_itr_)
      break;
    if (timeout_sec_ != steady_clock::duration::zero() &&
        duration_cast<duration<double>>(steady_clock::now() - start) >= timeout_sec_)
    {
      break;
    }
  }

  elapsed = duration_cast<duration<double>>(steady_clock::now() - start);
  num_iterations = iterations;

  for (auto &state : replicas)
  {
    state.current.recompute();
    state.best_correct.recompute();
    state.best_yet.recompute();
  }
}

StatisticsCallbackData Search::get_statistics() const {
  return {move_statistics, num_iterations, elapsed, transform_};
}

void Search::stop() {
  give_up_now = true;
}

void Search::configure(const Cfg& target, CostFunction& fxn, SearchState& state, vector<TUnit>& aux_fxn) const {
  state.current.recompute();
  state.best_yet.recompute();
  state.best_correct.recompute();

  // add dataflow information about function call targets
  for (const auto& fxn : aux_fxn) {
    const auto& code = fxn.get_code();
    const auto& lbl = fxn.get_leading_label();
    TUnit::MayMustSets mms = {
      code.must_read_set(),
      code.must_write_set(),
      code.must_undef_set(),
      code.maybe_read_set(),
      code.maybe_write_set(),
      code.maybe_undef_set()
    };
    state.current.add_summary(lbl, fxn.get_may_must_sets(mms));
  }

  state.current_cost = fxn(state.current).second;
  state.best_yet_cost = fxn(state.best_yet).second;
  state.best_correct_cost = fxn(state.best_correct).second;
  state.success = false;

  // @todo -- Let's move these invariants into SearchState
  // Redirecting the user here to reason about this seems like an opportunity for error

  // Invariant 3: Best correct should be correct with respect to target
  assert(fxn(state.best_correct).first);
  // Invariant 4: Best yet should be less than or equal to correct cost
  assert(state.best_yet_cost <= state.current_cost);
}



} // namespace stoke
