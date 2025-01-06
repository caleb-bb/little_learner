;; BEGIN CHAPTER ONE CODE
(define pie 3.14)
(define a-radius 8.4)
(define an-area
  (* pie
     (* a-radius a-radius)))

(define area-of-circle
(lambda (r)
  (* pie
     (* r r))))

(define area-of-rectangle
  (lambda (width)
    (lambda (height)
      (* width height))))

 (define double-result-of-f
   (lambda (f)
     (lambda (z)
       (* 2 (f z)))))

(define add3
  (lambda (x)
    (+ 3 x)))

(define abs
  (lambda (x)
    (cond
     ((< x 0) (- 0 x))
     (else x))))

(define silly-abs
  (lambda (x)
    (let ((x-is-negative (< x 0)))
      (cond
       (x-is-negative (- 0 x))
       (else x)))))

(define remainder
  (lambda (x y)
    (cond
     ((< x y) x)
     (else (remainder (- x y) y)))))

(define add
  (lambda (n m)
    (cond
     ((zero? m) n)
     (else (add1 (add n (sub1 m)))))))

;; END CHAPTER ONE CODE
