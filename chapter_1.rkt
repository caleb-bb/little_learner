#lang racket
(require malt)
;; BEGIN CHAPTER 1 CODE

;; Notice that this function can return any two dimensional line whatsoever. You
;; might ask, "But why write it like this? Why not just write a function that takes
;; three arguments and return the bloody line?" The answer is that, in the future,
;; we want to guess the correct line for a set of points. This higher-order
;; function takes a value for x, and then returns a curried function that wants w
;; and b. By separating the arguments like this, we've created a structure that can
;; generate any line we want. We can use this later on for fitting lines to
;; scatterplots of data, which will be fundamental for AI.
;;
;; x is known as an "argument of line", while w and b are "parameters of line".
(define line-ugly
  (lambda (x)
    (lambda (w b)
      (let ((y (+ (* w x) b)))
        y))))

(define line-better
  (lambda (x)
    (lambda (w b)
      (+ (* w x) b))))

(define line
  (lambda (x)
    (lambda (theta)
      (+ (* (ref theta 0) x) (ref theta 1)))))

;; A example of the curried version
(define line-8 (line-better 8))
(line-8 2 3)

;; Steps of evaluation for line-8
;; 0. (line-8 2 3)
;; 1. ((line-better 8) 2 3)
;; 2. ((lambda (w b)
;;      (+ (* w 8) b))
;;      2 3)
;; 3. (+ (* 2 8) 3)
;; 4. (+ 16 3)
;; 5. 19

;; The domain of our function is all of the given values for its arguments of line
;; The parameters of line are to be calculated in order to map the given arguments of line to the domain

(define line-xs (tensor 2.0 1.0 4.0 3.0))
(define line-ys (tensor 1.8 1.2 4.2 3.3))

;; Same-as chart for line
;; 1. ((line 7.3) (list 1.0 0.0))
;; 2. ((lambda (theta)
;;  (+ (* (ref theta 0) 7.3) (ref theta 1))
;;  (list 1.0 0.0)))
;; 3. (+ (* 1.0 7.3) 0.0)
;; 4. 7.3

;; END CHAPTER 1 CODE
