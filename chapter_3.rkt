#lang racket
(require malt)

(define line
  (lambda (x)
    (lambda (theta)
      (+ (* (ref theta 0) x) (ref theta 1)))))

(define sum
  (lambda (t)
    (summed t (sub1 (tlen t)) 0.0)))

(define summed
  (lambda (t i a)
    (cond
      ((zero? i) (+ (tref t 0) a))
      (else (summed t (sub1 i) (+ (tref t i) a   ))))))

(define loss
  (lambda (actual)
    (lambda (arguments parameters)
      (sum (sqr (- actual ((line arguments) parameters)))))))

(define l2-loss-pass1
  (lambda (xs ys)
    (lambda (theta)
     (let ((pred-ys ((line xs) theta)))
       (sum (sqr (- ys pred-ys)))))))

;; (define l2-loss-pass2
;;   (lambda (line)
;;     (lambda (xs ys)
;;       (lambda (theta)
;;         (let ((pred-ys ((line xs) theta)))
;;           (sum (sqr (- ys pred-ys))))))))

(define l2-loss
  (lambda (target)
    (lambda (xs ys)
      (lambda (theta)
        (let ((pred-ys ((target xs) theta)))
          (sum (sqr (- ys pred-ys))))))))

;; This is an "expectant function", because it is expecting a data set as
;; arguments. l2-loss, in itself, is not an expectant function because it takes a
;; function as a formal. It must be curried before it can be considered an
;; expectant function.
;;
;; 1. (l2-loss line)
;; 2. (lambda (xs ys)
;;      (lambda (theta)
;;        (let ((pred-ys ((line xs) theta)))
;;          (sum (sqr (- ys pred-yd))))))
;;
;; If we curry again, giving it a data set of xs and ys, it  goes from an
;; expectant function to an "objective function", meaning it awaits a theta as its
;; argument and returns a scalar representing the loss for that theta. Like so:
;;
;; 1. ((l2-loss line) line-xs line-ys)
;; 2. (lambda (theta)
;; (let ((pred-ys ((line line-xs) theta)))
;;   (sum
;;    (sqr
;;     (- line-ys pred-ys)))))
;;
;; Let's see the scalar:
;; 1. (((l2-loss line) line-xs line-ys)
;;      (list 0.0 0.0))
;;
;; 2. (sum
;;      (sqr
;;        (- ys
;;          ((line line-xs)
;;            (list 0.0 0.0)))))
;;
;; 3. (sum (sqr (- ys (tensor 0.0 0.0 0.0 0.0))))
;;
;; 4. (sum (sqr (tensor 1.8 1.2 4.2 3.3)))
;;
;; 5. 33.21

(define line-xs (tensor 2.0 1.0 4.0 3.0))
(define line-ys (tensor 1.8 1.2 4.2 3.3))
(define theta (list 0.0 0.0))

;; (l2-norm tensort) = (euclidean-distance tensore) = tensor -> sqr -> sum -> sqrt

;; The loss function determines how far we are from the desired result.
;; Supopse we change theta by some small amount. In that case, the loss will
;; also change by a small amount. So if our first loss is L1 and the second is L2,
;; and we changed theta by n, then (L2-L1)/n is our "rate of change"
;; This is our old friend of calculus, the derivative. Good old Dy/Dx
;;
;; Keep in mind that there are many variables here, because theta can have any
;; number of members. So what does that mean? It means that this is essentially
;; multivariate calculus, with the derivative here being the derivative
;; specifically with respect to theta_0
;;
;; Once we have a rate of change, we take a very small scalar called the
;; "learning rate" and multiply it by the derivative. This is the amount that we
;; use to revise our theta. It's also known as step size.
;;
;; The initial small revision to find the rate of change need not be part of the
;; first step.
;;
;; The Law Of Revision: new theta_0 = theta_0 - ((learning rate/step size) * (derivative with respect to theta_0))
;;
;; Keep in mind that the rate of change, itself, changes with each step.
