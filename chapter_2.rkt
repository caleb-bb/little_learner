#lang racket
(require malt)

(define rank-bad
  (lambda (t)
    (cond
      ((scalar? t) 0)
      (else (add1 (rank (tref t 0)))))))

(define shape
  (lambda (t)
    (cond
      ((scalar? t) (list))
    (else (cons (tlen t) (shape (tref t 0)))))))

(define ranked
  (lambda (t a)
    (cond
      ((scalar? t) a)
      (else (ranked (tref t 0) (add1 a)))
      )))

(define rank
  (lambda (t)
    (ranked t 0)))

;; The Law of Simple Accumulator Passing
;; In a simple accumulator passing functrion dewfinition, every recursive
;; function invocation is unwrapped, and the definition has at most one argument
;; that does not change. An argument that changes troward a true base test. And
;; another that accumulates a result.
