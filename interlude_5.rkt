#lang racket
(require malt/base)
(require malt)
;; tmap is an extension of map to tensors.
;; 1. (tmap add1 [3 5 4])
;; 2. [(add1 3) (add1 5) (add1 4)]
;; 3. [4 6 5]

;; Let's extend sqrt to tensors!
;;
;; (define (of-rank? n t)
;;   (= (rank t) n))

(define of-rank?
  (lambda (n t)
    (cond
      [(zero? n) (scalar? t)]
      [(scalar? t) #f]
      [else
       (and (tensor? t) ; Check if it's a tensor first
            (of-rank? (sub1 n) (tref 0 t)))])))

(define extended-sqrt
  (lambda (t)
    (cond
      [(of-rank? 0 t) (sqrt-0 t)]
      [else (tmap sqrt t)])))

(define ext1
  (lambda (f)
    (lambda (t)
      (cond
        [(of-rank? 0 t) (f t)]
        [else (tmap (ext1 f) t)]))))

;; (define sqrt (ext1 sqrt-0 0))
;; (define zeroes (ext1 (lambda (x) 0.0) 0))
;; (define sum (ext1 sum-1 1))

;; (define flatten (ext1 flatten-2 2))

;; (flatten (tensor (tensor 1 2) (tensor 3 4)))

(define rank>
  (lambda (t u)
    (cond
      [(scalar? t) #f]
      [(scalar? u) #t]
      [else (rank> (tref t 0) (tref u 0))])))

(define of-ranks?
  (lambda (n t m u)
    (cond
      [(of-rank? n t) (of-rank? m u)]
      [else #f])))

(define desc-t (lambda (g t u) (tmap (lambda (et) (g et u)) t)))
(define desc-u (lambda (g t u) (tmap (lambda (eu) (g t eu)) u)))

(define desc
  (lambda (g n t m u)
    (cond
      [(of-rank? n t) (desc-u g t u)]
      [(of-rank? m u) (desc-t g t u)]
      [(= (tlen t) (tlen u)) (tmap g t u)]
      [(rank> t u) (desc-t g t u)]
      [else (desc-u g t u)])))

;; For this function, t and u are tensors and n and m are integers. If t has rank n
;; and u has rank  m, then we invoke f on t and u. Otherwise, we descend into the tensors.
(define ext2
  (lambda (f n m)
    (lambda (t u)
      (cond
        [(of-ranks? n t m u) (f t u)]
        [else (desc (ext2 f n m) n t m u)]))))

;; "Descend into the tensors" here means that we invoke desc/4. To understand that
;; function, we have to understand desc-t and desc-u.
;;
;; desc-t and desc-u are, mathematically, the same function. Naming them
;; differently is a matter of convenience, which we will see if we look at the
;; alternative. If we only defined onne such function, then the call to cond in
;; ext2 would look slightly different.
;;
;; We would simply be calling desc-x with the arguments in a different order.
;; This, however, would be a little confusing to read. So, for our own convenience,
;; we define two different functions.
;;
;; Let there be a scalar, a tensor of rank greater than 0, and a binary
;; operation. If we want to map the binary operation over these two, we do it like
;; this: we take the function, we curry it so that its first term is always the
;; scalar, and then we apply that curried function to every term in the higher-rank
;; tensor. If that tensor has a rank greater than one, then we'll descend into it
;; until we'r applying that binary function to every scalar.
;;
;; SECOND ATTEMPT AT EXPLAINING
;; Let a binary scalar operation (BSO) be any commutative operation that takes two scalars and returns a scalar.
;; Let a binary tensor operation (BTO) be a generalization of a BSO that:
;;    1. Takes two tensors as arguments
;;    2. Takes tensors of any rank
;;    3. Can take two tensors of differing rank
;;    4. Returns a tensor of rank R, where R is the rank of the argument with the highest rank.
;;
;; How do we effect this generalization? Simple. Let O be a BTO We stipulate that:
;;    1. For any scalar s and any tensor of rank > 0 t, O(s, t) is a tensor where every scalar T_tn == O(s, T_tn)
;;    2. For any two tensors t1 and t2 of equal rank > 0, O(t1, t2) is a tensor t3 whose scalars are the output of the corresponding scalars in the arguments operated upon pairwise
;;    3. For any other pair of tensors with unequal ranks t1 and t2 such that rank(t1) < rank(t2), O(t1, t2) == map(O, t1, t2).
;; This happens recursively until condition 1 or 2 is triggered.
;;

(define +-2 (ext2 + 0 0))
(define *-2 (ext2 * 0 0))
(define *-two (ext2 * 0 0))

(define sqr (lambda (t) (* t t)))
