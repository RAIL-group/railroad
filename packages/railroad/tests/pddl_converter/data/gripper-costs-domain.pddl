;; Gripper variant with :action-costs, including a function-valued move cost.
(define (domain gripper-costs)
  (:requirements :strips :typing :action-costs)
  (:types room ball)
  (:predicates (at-robby ?r - room)
               (at ?b - ball ?r - room)
               (carry ?b - ball)
               (hand-empty))
  (:functions (total-cost) - number
              (move-cost ?from - room ?to - room) - number)

  (:action move
     :parameters (?from - room ?to - room)
     :precondition (at-robby ?from)
     :effect (and (not (at-robby ?from)) (at-robby ?to)
                  (increase (total-cost) (move-cost ?from ?to))))

  (:action pick
     :parameters (?b - ball ?r - room)
     :precondition (and (at ?b ?r) (at-robby ?r) (hand-empty))
     :effect (and (carry ?b) (not (at ?b ?r)) (not (hand-empty))
                  (increase (total-cost) 1)))

  (:action drop
     :parameters (?b - ball ?r - room)
     :precondition (and (carry ?b) (at-robby ?r))
     :effect (and (at ?b ?r) (hand-empty) (not (carry ?b))
                  (increase (total-cost) 1))))
