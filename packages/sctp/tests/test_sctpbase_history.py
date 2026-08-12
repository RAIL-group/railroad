import pytest
from basesctp import base_pomdpstate

def test_sctpbase_history():
   start = 1
   im_node1 = 2
   # im_node2 = 3
   goal = 4
   history1 = base_pomdpstate.History()
   history2 = base_pomdpstate.History()
   history3 = base_pomdpstate.History()
   action1 = base_pomdpstate.Action(start_node=start, target_node=im_node1)

   EnventOutcome = base_pomdpstate.EventOutcome

   history1.add_history(action=action1, outcome=EnventOutcome.BLOCK)
   history2.add_history(action=action1, outcome=EnventOutcome.BLOCK)
   history3.add_history(action=action1, outcome=EnventOutcome.TRAV)
   assert history1 == history2
   assert history1 != history3


   action2 = base_pomdpstate.Action(start_node=im_node1, target_node=goal)

   history1.add_history(action=action2, outcome=EnventOutcome.TRAV)
   history2.add_history(action=action2, outcome=EnventOutcome.TRAV)
   assert history1 == history2

   history2.add_history(action=action1, outcome=EnventOutcome.TRAV)

   assert history1 != history2
