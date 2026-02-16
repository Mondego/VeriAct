let start_solver () = 
  print_string "ML solver is starting...";
  print_newline;;

let _ = Callback.register "start_solver" start_solver

let set_flags flags = 
  print_string "ML: setting resource flags: ";
  print_string flags;
  print_newline;;

let _ = Callback.register "set_flags" set_flags

let declaration xr_sgt =
  print_string "ML: declaring signature: ";
  print_string xr_sgt;
  print_newline;;

let _ = Callback.register "declaration" declaration

let add_axiom xr_axiom =
  print_string "ML: adding axiom: ";
  print_string xr_axiom;
  print_newline;;

let _ = Callback.register "add_axiom" add_axiom

let add_assertion xr_asrt =
  print_string "ML: asserting term: ";
  print_string xr_asrt;
  print_newline;;

let _ = Callback.register "add_assertion" add_assertion

let backtrack n =
  print_string "ML: backtracking ";
  print_int n;
  print_string " assumptions.";
  print_newline;;

let _ = Callback.register "backtrack" backtrack

let query xr_q =
  print_string "ML: querying the VC: ";
  print_string xr_q;
  print_newline;;

let _ = Callback.register "query" query

let stop_solver () =
  print_string "ML: solver is stopping...";
  print_newline;;

let _ = Callback.register "stop_solver" stop_solver
