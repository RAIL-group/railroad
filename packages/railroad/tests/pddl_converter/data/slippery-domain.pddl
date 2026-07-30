;; Minimal PPDDL domain: picking up a block may slip.
(define (domain slippery-blocks)
  (:requirements :strips :probabilistic-effects)
  (:predicates (on-table ?b) (holding ?b) (hand-empty) (delivered ?b))

  (:action pickup
     :parameters (?b)
     :precondition (and (on-table ?b) (hand-empty))
     :effect (and (not (hand-empty))
                  (probabilistic
                     7/10 (and (holding ?b) (not (on-table ?b)))
                     0.3  (hand-empty))))

  (:action deliver
     :parameters (?b)
     :precondition (holding ?b)
     :effect (and (not (holding ?b)) (hand-empty) (delivered ?b))))
