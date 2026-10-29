# Physical_AI_Fleet

Event-driven Physical AI fleet extension for Isaac Sim. Go2 / Spot quadrupeds and H1 / G1 humanoids
walk with RL locomotion policies (planner -> path follower -> policy at 200 Hz); AMRs and digital twins
are posed kinematically. Optional environment USD.

Pipeline: camera + sensors -> gauge pipeline -> Cosmos Reason -> validated fleet event -> coordinator ->
Nemotron on Nebius Token Factory -> task assignment -> safety gate -> skill -> verify.
