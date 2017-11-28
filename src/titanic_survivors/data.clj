(ns titanic-survivors.data
  (:require
    [incanter.core :as i]
    [incanter.io :as iio]))


;; Data In / Out

;; File Paths
(def file-paths {:train       "./data/train.csv"
                 :test        "./data/test.csv"
                 :predictions "./data/predictions.csv"})

(defn load-data
  "Reads in data from file."
  [file-path]
  (iio/read-dataset file-path
                    :header true
                    :keyword-headers true))

(defn save-data
  "Saves data to a file."
  [file-path data]
  (i/save data file-path))

(defn save-predictions
  "Save the survival predictions to a .csv file"
  [predictions]
  (save-data (:predictions file-paths) predictions))

;; Dataset manipulation

(defn add-dummy
  "Creates a dummy variable for a given column in a dataset.
  When the value in the input column equals value, the dummy
  column will contain a 1, else 0."
  [column-name from-column value dataset]
  (i/add-derived-column column-name
                        [from-column]
                        #(if (= % value) 1 0)
                        dataset))

(defn update
  [m k f]
  (update-in m [k] f))

(defn add-normalized-column
  "Performs mean-normalization on from-column into column name."
  [column-name from-column dataset]
  (let [col-data                (filter some? (i/$ from-column dataset))
        {:keys [min max sum n]} (reduce (fn [accum x] (-> accum
                                      (update :min #(min % x))
                                      (update :max #(max % x))
                                      (update :n   inc)
                                      (update :sum #(+ % x))))
                                        {:min 0
                                         :max 0
                                         :sum 0
                                         :n   0}
                                        col-data)
        mean (/ sum n)
        f (fn [x] (if x
                    (/ (- x mean) (- max min))
                    (/ mean (- max min))))]
    (i/add-derived-column column-name
                          [from-column]
                          f
                          dataset)))

(defn replace-nils-with-mean
  "Replaces the nil values in a column with the mean of that column."
  [from-column dataset]
  (let [col-data                (filter some? (i/$ from-column dataset))
        {:keys [min max sum n]} (reduce (fn [accum x] (-> accum
                                                          (update :min #(min % x))
                                                          (update :max #(max % x))
                                                          (update :n   inc)
                                                          (update :sum #(+ % x))))
                                        {:min 0
                                         :max 0
                                         :sum 0
                                         :n   0}
                                        col-data)
        mean (/ sum n)
        f (fn [x] (if x
                    x
                    mean))]
    (i/transform-col dataset from-column f)))

(defn add-product-column
  "Adds a derived column from the product of the supplied columns."
  [column-name from-columns dataset]
  (i/add-derived-column column-name
                        from-columns
                        #(* %1 %2)
                        dataset))

(defn matrix-dataset
  "Converts the titanic data to a feature matrix, adding dummy columns
  for categorical variables (e.g. sex). Add a column for the bias term."
  [data-key]
  (->> (load-data (get file-paths data-key))
       (i/add-column :Bias (repeat 1.0))
       (add-dummy :DummyMaleFemale :Sex "male")
       (add-dummy :Dummy1Class :Pclass 1)
       (add-dummy :Dummy2Class :Pclass 2)
       (add-dummy :Dummy3Class :Pclass 3)
       (add-normalized-column :NormalizedAge :Age)
       (add-product-column :DummySexAge [:DummyMaleFemale :NormalizedAge])
       (i/$ (into [] (concat (case data-key
                               :train [:Survived]
                               :test  [:PassengerId])
                             [:Bias :DummyMaleFemale :Dummy1Class :Dummy2Class :Dummy3Class
                              :SibSp :NormalizedAge :DummySexAge])))
       (i/to-matrix)))

(defn age-categories
  "Splits the continuous :Age variable into three categories."
  [age]
  (cond
    (nil? age) "unknown"
    (< age 13) "child"
    :default   "adult"))

(defn fare-categories
  "Splits the continous :Fare variable into four categories."
  [fare]
  (cond
    (nil? fare)   "unknown"
    (<= fare 10)  "poor"
    (<= fare 100) "standard"
    :default      "wealthy"))