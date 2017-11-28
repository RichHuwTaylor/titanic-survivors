(ns titanic-survivors.statistics
  (:require
    [clojure.walk :refer [keywordize-keys]]
    [incanter.core :as i]))


;; Statistics

(defn frequency-table
  "Constructs a frequency table showing how counts in each group of
  the given dataset are distributed."
  [sum-column group-columns dataset]
  (->> (i/$ group-columns dataset)
       (i/add-column sum-column (repeat 1))
       (i/$rollup :sum sum-column group-columns)))

(defn frequency-map
  "Converts the dataset into a series of nested maps."
  [sum-column group-cols dataset]
  (let [f (fn [freq-map row]
            (let [groups (map row group-cols)]
              (->> (get row sum-column)
                   (assoc-in freq-map groups))))]
    (->> (frequency-table sum-column group-cols dataset)
         (:rows)
         (reduce f {})
         (keywordize-keys))))

(defn fatalities-by-sex
  [dataset]
  "Produces a map of fractions of fatalities from dataset by sex."
  (let [totals (frequency-map :count [:Sex] dataset)
        groups (frequency-map :count [:Sex :Survived] dataset)]
    {:male (/ (get-in groups [:male 0])
              (get totals :male))
     :female (/ (get-in groups [:female 0])
                (get totals :female))}))

(defn confusion-matrix
  "Produces a confusion matrix, showing classification of items in the
  training set. Splits into true positives, true negatives, false positives,
  and false negatives. Takes a vector of labels, ys, and a vector of
  predictions, y-hats."
  [ys y-hats]
  (let [classes   (into #{} (concat ys y-hats))
        confusion (frequencies (map vector ys y-hats))]
    (i/dataset (cons nil classes)
               (for [x classes]
                 (cons x
                       (for [y classes]
                         (get confusion [x y])))))))

(defn kappa-statistic
  "Takes a vector of labels, ys, and a vector of predictions, y-hats,
  and calculates the kappa statistic, the probability of random agreement
  between predicted and actual values."
  [ys y-hats]
  (let [n   (count ys)
        pa  (/ (count (filter true? (map = ys y-hats))) n)
        ey  (/ (count (filter zero? ys)) n)
        eyh (/ (count (filter zero? y-hats)) n)
        pe  (+ (* ey eyh)
               (* (- 1 ey)
                  (- 1 eyh)))]
    (/ (- pa pe)
       (- 1 pe))))

(defn information
  "Calculates the information associated with probability x."
  [x]
  (- (i/log2 x)))

(defn entropy
  "Calculates the entropy, the sum of the products of P(x) and I(P(x))."
  [xs]
  (let [n (count xs)
        f (fn [x]
            (let [p (/ x n)]
              (* p (information p))))]
    (->> (frequencies xs)
         (vals)
         (map f)
         (reduce +))))

(defn weighted-entropy
  "Calculates the weighted average entropy for the supplied groups."
  [groups]
  (let [n (count (apply concat groups))
        f (fn [group]
            (* (entropy group)
               (/ (count group) n)))]
    (->> (map f groups)
         (reduce +))))

(defn information-gain
  "Returns the information gain from grouping the data into groups."
  [groups]
  (- (entropy (apply concat groups))
     (weighted-entropy groups)))


