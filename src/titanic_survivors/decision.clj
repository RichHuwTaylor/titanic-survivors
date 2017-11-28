(ns titanic-survivors.decision
  (:require
    [titanic-survivors.statistics :as statistics]))

(defn map-vals [f coll]
  "Constructs a map of key value pairs by mapping over a collection
  of key value pairs, applying f to the values."
  (into {} (map (fn [[k v]] [k (f v)]) coll)))

(defn gain-for-predictor
  "Calculates the information gain by knowing predictor when trying
  to predict class-attr."
  [class-attr xs predictor]
  (let [grouped-classes (->> (group-by predictor xs)
                             (vals)
                             (map (partial map class-attr)))]
    (statistics/information-gain grouped-classes)))

(defn best-predictor
  "Takes a collection of predictors and returns the one with
  the highest information gain."
  [class-attr predictors xs]
  (let [gain (partial gain-for-predictor class-attr xs)]
    (when (seq predictors)
      (apply max-key gain predictors))))

(defn modal-class
  "Finds the most common class, given a sequence of data."
  [classes]
  (->> (frequencies classes)
       (apply max-key val)
       (key)))

(defn decision-tree
  "Partially recursive function for building a decision tree.
  If the entropy is zero, returns the first of classes, otherwise
  branch on the best predictor and recursively call decision-tree
  with the remaining predictors, group by best-predictor, and call
  partially applied tree-branch function on each group. Wraps the
  result in a vector."
  [class-attr predictors xs]
  (let [classes (map class-attr xs)]
    (if (zero? (statistics/entropy classes))
      (first classes)
      (if-let [predictor (best-predictor class-attr
                                         predictors xs)]
        (let [predictors  (remove #{predictor} predictors)
              tree-branch (partial decision-tree
                                   class-attr predictors)]
          (->> (group-by predictor xs)
               (map-vals tree-branch)
               (vector predictor)))
        (modal-class classes)))))


(defn tree-classify
  "Takes a decision tree and a test, and returns the classification
  for that test."
  [model test]
  (if (vector? model)
    (let [[predictor branches] model
          branch (get branches (get test predictor))]
      (recur branch test))
    model))