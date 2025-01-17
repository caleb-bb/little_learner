;; d
#lang racket
(require malt)

;; THE LAW OF REVISIONS
;; As long as we mek sure that gradient-descent accepts an initial theta and
;; results in a well-fitted theta, any reasonable way of revising it from the first
;; to the last revision is okay.

;; (define useless-map (theta)
;;   (map (lambda (p)
;;          (list p))
;;        theta))

;; A singleton is a list with one member
;; Whenever a parameter is wrappd in a list, we refer to it as an accompanies parameter

(define (lonely-i theta)
  (list theta))

(define (lonely-d theta)
  (ref theta 0))

(define (lonely-u theta g)
  (list (- (ref theta 0) (* alpha g))))

; i, d, u stand, respectively, for inflate, deflate, and update.

(define plane-xs
  (tensor (tensor 1.0 2.05)
          (tensor 1.0 3.0)
          (tensor 2.0 2.0)
          (tensor 2.0 3.9)
          (tensor 3.0 6.13)
          (tensor 4.0 8.09)))

(define plane-ys (tensor 13.99 15.99 18.0 22.4 30.2 37.94))

;; (define penultimate-gradient-descent
;;   (lambda (inflate deflate update)
;;     (lambda (obj theta)
;;       (let ([f (lambda (big_theta) (update big_theta (gradient-of obj (deflate big_theta))))])
;;         (deflate (revise f revs (inflate theta)))))))

(define gradient-descent
  (lambda (inflate deflate update)
    (lambda (obj theta)
      (let ([f (lambda (big_theta) (map update big_theta (gradient-of obj (map deflate big_theta))))])
        (map deflate (revise f revs (map inflate theta)))))))

(define lonely-gradient-descent (gradient-descent lonely-i lonely-d lonely-u))

(define try-plane
  (lambda (a-gradient-descent)
    (with-hypers ((revs 150000) (alpha 0.0001) (batch-size 4))
                 (a-gradient-descent (sampling-obj (l2-loss plane) plane-xs plane-ys)
                                     (list (tensor 0.0 0.0) 0.0)))))

(define naked-i (lambda (p) (let ([big_p p]) big_p)))
(define naked-d (lambda (big_p) (let ([p big_p]) p)))
(define naked-u (lambda (big_p g) (- big_p (* alpha g))))

(define naked-gradient-descent (gradient-descent naked-i naked-d naked-u))
;; This last definition of gradient-descent shows us the naked structure of
;; gradient-descent. It returns a function that wants an objective function and a
;; data set. And what happens if you give it that?
;;
;; Simple. First, it defines a lambda, f, which takes a big_theta (an accompanied theta).
;; f begins by mapping deflate over the big_theta, to turn its parameters into
;; unaccompanied parameters. Then, it feeds those unaccompanies parameters to
;; gradient-of along with the objective function, which produces a gradient (list
;; of derivatives). Then, f maps the update function over big_theta and the list of
;; derivatives.
;; Finally, our naked gradient-descent creates a list by mapping inflate over theta, turning theta into a set of accompanied parameters to be passed to f. This list is passed, along with f, into revise.
;; revise applies f to the inflated theta (that is, the big_theta) rev times. Then it maps deflate over that whole list, to turn it all back into the same theta shape we knew at the beginning

;; our inflate, deflate, and update functions are concerned with only one parameter
;; and its representations, and are not directly concerned with either theta or
;; big_theta inflate turns lone parameters into accompanied parameters deflate
;; converts accompanies parameters into lone ones
