import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import reciprocal
import matplotlib.pyplot as plt
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import average_precision_score, ConfusionMatrixDisplay, mean_absolute_error
import sys
import pandas_ta as ta
import os
from scipy.stats import norm
import traceback
import json
from sklearn.metrics import jaccard_score # Jacard Score looks at the similarity only when we trade
import copy

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
os.environ["CUDA_PATH"] = cuda_path
os.environ["PATH"] = cuda_path + r"\bin;" + os.environ["PATH"]
import cupy as cp

class Differential_Evolution:
    def __init__(self, X , y, fitness_function = average_precision_score, pop_size = 200, overint = 1, mutation_rate = 0.1, 
                 shrinkage_constant = None, max_generations = 100, model = XGBClassifier, parameter_dic = None, fixed_parameter_dic = None,
                 ohlc = None, use_walk_forward = False, walk_forward_train_size = 0.5, walk_forward_step_size = 0.1):

        self.individual_dic = {}
        self.fixed_parameter_dic = fixed_parameter_dic if fixed_parameter_dic is not None else {} 
        self.pop_size = pop_size
        self.overint = overint * pop_size
        self.mutation_rate = mutation_rate
        self.shrinkage_constant = shrinkage_constant
        self.X = cp.array(X.values, dtype=cp.float32)
        self.y = cp.array(y.values, dtype=cp.float32)
        self.model = model
        self.parameter_dic = parameter_dic # The hyper parameters to be optimized along with their ranges as a dictionary
        self.fitness_function = fitness_function # This function evaluates the fitness of each individual
        self.individual_fitness = [None for _ in range(0, self.pop_size)]
        self.individual_returns = [None for _ in range(0, self.pop_size)]
        self.current_predictions = None
        self.best_fitness_current_generation = None
        self.ran_elitism = False

        self.use_walk_forward = use_walk_forward

        self.max_generations = max_generations
        self.model_type = 'classifier' if is_classifier(self.model) else 'regressor'

        self.results_dir = "de_optimization_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)


        if fitness_function == 'Martin Ratio' and isinstance(ohlc, pd.DataFrame) == False:
            print("To work with the Martin Ratio Fitness function pass a dataframe under 'ohlc' with col ")

        if self.model_type == 'classifier':
            self.ohlc = ohlc
            self.ohlc['return'] = self.ohlc['premium'] - self.ohlc['abs_daily_return']

        self.population_predictions = np.zeros((self.pop_size, self.X.shape[0]))

        if use_walk_forward:
            self.use_walk_forward = use_walk_forward
            self.walk_forward_train_size = walk_forward_train_size
            self.walk_forward_step_size = walk_forward_step_size
            self.population_predictions = np.zeros((self.pop_size, int(self.X.shape[0] * self.walk_forward_train_size)))


        # Seed the initial population
        for ind in range(0, self.pop_size + self.overint):
                
            if ind < self.pop_size: # Seed the initial group of individuals
                self.individual_dic[ind] = self.fixed_parameter_dic.copy() # Eah individual is represented as a dictionary and it is initialized with the fixed parameters

                for key, value in self.parameter_dic.items():
                    # All the values will use a log-uniform distribution i.e. the distane from the smallest to the largest value is uniform in log scale so it encompasses all types of magnitudes and spends more time on 10 to 100 than on 100 to 1000 if the given range was 10 to 1000

                    hyper_parameter = 10**np.random.uniform(np.log10(value[0]), np.log10(value[1])) # This generates the hyper parameter in log-space
                    
                    if isinstance(value[0], int) and isinstance(value[1], int): # If both the constraints are integers we round otherwise  we leave it as a continious variable
                        hyper_parameter = int(np.round(hyper_parameter))

                    self.individual_dic[ind][key] = hyper_parameter# Adds the random hyper parameter to the 'individual'

                model_ = self.model(**self.individual_dic[ind]) # Intialize the model with the hyper parameters of the individual
                model_.fit(self.X, self.y)

                if isinstance(self.fitness_function, str) == False: # Calculate the fitness using an inbuilt metric if the fitness function is not a string otherwise call the other type
                
                    if self.model_type == 'classifier':

                        if self.use_walk_forward == False: # If NOT a walkforward
                            fitness_, returns = self.calculate_fitness_inbuilt_metric(model_)
                        else: # If a walkforward
                            fitness_, returns = self.calculate_fitness_inbuilt_metric_walkforward(self.individual_dic[ind])

                    else:
                        if self.use_walk_forward == False:
                            fitness_ = self.calculate_fitness_inbuilt_metric(model_)
                        else:
                            fitness_ = self.calculate_fitness_inbuilt_metric_walkforward(self.individual_dic[ind])
                
                else:
                    if self.use_walk_forward == False:
                        fitness_, returns = self.calculate_fitness_martin_ratio(model_)
                    else:
                        fitness_, returns = self.calculate_fitness_martin_ratio_walkforward(self.individual_dic[ind])



                self.individual_fitness[ind] = fitness_ # Store the fitness of the individual
                self.individual_returns[ind] = pd.Series(returns).sum() if self.model_type == 'classifier' else None
                self.population_predictions[ind, :] = self.most_recent_oos_predictions
                self.most_recent_oos_predictions = None

            else: # This is the overint part, where additional individuals are created and we check if they have better fitness than the worst indivudal
                
                if self.model_type == 'classifier':
                    individual_, fitness_, returns = self.initialize_individual()
                else:
                    individual_, fitness_ = self.initialize_individual()

                worst_fitness = np.max(self.individual_fitness)


                if fitness_ < worst_fitness: # If the new individual is better than the worst individual we replace it
                    worst_fitness_index = np.argmax(self.individual_fitness)
                    self.individual_dic[worst_fitness_index] = individual_
                    self.individual_fitness[worst_fitness_index] = fitness_
                    self.individual_returns[worst_fitness_index] = pd.Series(returns).sum() if self.model_type == 'classifier' else None
                    self.population_predictions[worst_fitness_index, :] = self.most_recent_oos_predictions

            print(self.individual_fitness)
        
        for generation in range(0, self.max_generations):

            # Get the minimum (best) fitness of the current generation
            self.best_fitness_current_generation = np.min(self.individual_fitness)

            self.ran_elitism = False

            self.run_generation(generation = generation)
            print(f"Generation {generation} completed. Best fitness: {np.min(self.individual_fitness)}")

            hamming_score = self.compute_hamming_similarity()
            jaccard_score = self.compute_jaccard_similarity()

            print(f'--> Population Hamming Similarity: {hamming_score}')
            print(f'--> Population Jaccard Similarity: {jaccard_score}')


            self.run_best_individual_optimization(similarity_score = jaccard_score, sigma_base = 0.1)
            print(f"Optimization {generation} completed. Best fitness: {np.min(self.individual_fitness)}")
            # Save the best performer of the current generation
            self.save_best_params(generation)


    def calculate_martin_ratio(self, returns):

        returns = pd.Series(returns) # Convert to a pandas series so we can use cumprod and cummax
 
        wealth_index = returns.cumsum() # Calculate the cummulative return

        previous_peaks = wealth_index.cummax() # Calculate the max point before every index
        drawdowns = (wealth_index - previous_peaks) / previous_peaks # Calculate the drawdown at each point
        ulcer_index = np.sqrt(np.mean(drawdowns**2)) # Calculate the Ulcer Index from the Drawdown
        total_return = wealth_index.iloc[-1] - 1


        # Handle division by zero, by giving such a high Martin Ratio that 
        if ulcer_index == 0:
            return 0

        martin_ratio = total_return / ulcer_index

        return -1 * martin_ratio # Return - martin_ratio to make it consistent with the approach of minimizing

    def calculate_fitness_martin_ratio(self, model_): # To be used if we are working with the martin ratio as the fitness function

        # We need to differentiate between classification and regression models
        if self.model_type == 'classifier':

            y_pred = model_.predict(self.X)

            self.most_recent_oos_predictions = y_pred

            y_ = self.ohlc['return'].values.tolist() # Get the realized returns of the option strategy
            returns = np.where(y_pred == 1, y_, 0)
            fitness = self.calculate_martin_ratio(returns= returns)
            
            return fitness, returns

        elif is_regressor(model_):
            pass
        

    def calculate_fitness_inbuilt_metric(self, model_): # The fitness will always be returned as negative, so that we minimize i.e. lower fitness is better to make it consistent across all metrics

        if self.fitness_function is average_precision_score:
            y_pred_proba = model_.predict_proba(self.X)[:, 1]
            fitness = -1 * self.fitness_function(self.y.get(), y_pred_proba)

            
            y_pred = model_.predict(self.X)
            y_ = self.ohlc['return'].values.tolist()
            returns = np.where(y_pred == 1, y_, 0)

            self.most_recent_oos_predictions = y_pred

            '''
            disp = ConfusionMatrixDisplay.from_predictions(self.y.get(), y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.show()
            '''
            return fitness, returns
            

        if self.fitness_function is mean_absolute_error:
            y_pred = model_.predict(self.X)
            self.most_recent_oos_predictions = y_pred
            fitness = self.fitness_function(self.y.get(), y_pred)

            return fitness


    def calculate_fitness_inbuilt_metric_walkforward(self, model_params): # A walk forward version of the inbuilt metric fitness function

        n_samples = self.X.shape[0]
        train_size = int(n_samples * self.walk_forward_train_size)
        step_size = int(n_samples * self.walk_forward_step_size)

        all_oos_returns = []
        all_oos_predictions = []
        for start_idx in range(0, n_samples - train_size, step_size):
                train_end = start_idx + train_size
                test_end = min(train_end + step_size, n_samples)
                
                # Slicing CuPy arrays is O(1) memory (views)
                X_train_chunk = self.X[start_idx:train_end]
                y_train_chunk = self.y[start_idx:train_end]
                X_test_chunk = self.X[train_end:test_end]
                
                # Get the real market returns for the test period
                # We need the ohlc returns aligned with the test indices
                test_returns_actual = self.ohlc['return'].iloc[train_end:test_end].values
                
                # Initialize and fit model for this specific window
                # We use the parameters provided by the DE individual
                m = self.model(**model_params)
                m.fit(X_train_chunk, y_train_chunk)

                # Predict out-of-sample
                y_pred = m.predict(X_test_chunk)
                if hasattr(y_pred, 'get'):
                    all_oos_predictions.extend(y_pred.get().tolist())
                else:
                    all_oos_predictions.extend(y_pred.tolist())
                
                if self.model_type == 'classifier': # If a classifier we also need to include the returns
                    # Strategy: if prediction is 1, take return, else 0
                    oos_window_returns = np.where(y_pred == 1, test_returns_actual, 0)
                    all_oos_returns.extend(oos_window_returns)

        # Calculate the inbuilt metric on the concatenated out-of-sample returns
        if len(all_oos_returns) == 0 and self.model_type == 'classifier':
                return 0, 0
        
        self.most_recent_oos_predictions = all_oos_predictions

        if self.model_type == 'classifier':
            return -1 * self.fitness_function(self.y.get()[train_size:], all_oos_predictions), all_oos_returns
        else:
            return - 1 * self.fitness_function(self.y.get()[train_size:], all_oos_predictions)


    def calculate_fitness_martin_ratio_walkforward(self, model_params):
        n_samples = self.X.shape[0]
        train_size = int(n_samples * self.walk_forward_train_size)
        step_size = int(n_samples * self.walk_forward_step_size)

        all_oos_returns = []
        all_oos_predictions = []

        for start_idx in range(0, n_samples - train_size, step_size):
                train_end = start_idx + train_size
                test_end = min(train_end + step_size, n_samples)
                
                # Slicing CuPy arrays is O(1) memory (views)
                X_train_chunk = self.X[start_idx:train_end]
                y_train_chunk = self.y[start_idx:train_end]
                X_test_chunk = self.X[train_end:test_end]
                
                # Get the real market returns for the test period
                # We need the ohlc returns aligned with the test indices
                test_returns_actual = self.ohlc['return'].iloc[train_end:test_end].values
                
                # Initialize and fit model for this specific window
                # We use the parameters provided by the DE individual
                m = self.model(**model_params)
                m.fit(X_train_chunk, y_train_chunk)
                
                # Predict out-of-sample
                y_pred = m.predict(X_test_chunk)
                
                # Strategy: if prediction is 1, take return, else 0
                oos_window_returns = np.where(y_pred == 1, test_returns_actual, 0)
                all_oos_returns.extend(oos_window_returns)
                all_oos_predictions.extend(y_pred)

        # Calculate the Martin Ratio on the concatenated out-of-sample returns
        if len(all_oos_returns) == 0:
                return np.nan
                
        self.most_recent_oos_predictions = all_oos_predictions
        return self.calculate_martin_ratio(all_oos_returns), all_oos_returns

    def initialize_individual(self): # A function that can be recalled to create an individual on demand, returns the individual and his fitnes

        individual = self.fixed_parameter_dic.copy()

        for key, value in self.parameter_dic.items():
            # All the values will use a log-uniform distribution i.e. the distane from the smallest to the largest value is uniform in log scale so it encompasses all types of magnitudes and spends more time on 10 to 100 than on 100 to 1000 if the given range was 10 to 1000

            hyper_parameter = 10**np.random.uniform(np.log10(value[0]), np.log10(value[1])) # This generates the hyper parameter in log-space
                    
            if isinstance(value[0], int) and isinstance(value[1], int): # If both the constraints are integers we round otherwise  we leave it as a continious variable
                hyper_parameter = int(np.round(hyper_parameter))

            individual[key] = hyper_parameter# Adds the random hyper parameter to the 'individual'

        model_ = self.model(**individual) # Intialize the model with the hyper parameters of the individual
        model_.fit(self.X, self.y)

        if isinstance(self.fitness_function, str) == False: # Calculate the fitness using an inbuilt metric if the fitness function is not a string otherwise call the other type
        
            if self.model_type == 'classifier': # If the model is a classifier we can get the returns of the trading strategy without any additional hyperparameters
                
                if self.use_walk_forward == False: # If NOT a walkforward
                    fitness_, returns = self.calculate_fitness_inbuilt_metric(model_)
                else: # If a walkforwards
                    fitness_, returns = self.calculate_fitness_inbuilt_metric_walkforward(individual)
                return individual, fitness_, returns
            
            else: # If the model is a regressor we just return the 
                
                if self.use_walk_forward == False:
                    fitness_ = self.calculate_fitness_inbuilt_metric(model_)
                else:
                    fitness_ = self.calculate_fitness_inbuilt_metric_walkforward(individual)
                return individual, fitness_, None
        
        else: # if we are using the martin ratio, we also get the returns
            if self.use_walk_forward == False:
                fitness_, returns = self.calculate_fitness_martin_ratio(model_)
            else:
                fitness_, returns = self.calculate_fitness_martin_ratio_walkforward(individual)
            
            return individual, fitness_, returns

    def compute_hamming_similarity(self): 
        # 1. Ensure P is definitely on GPU
        P = cp.asarray(self.population_predictions)
        N, L = P.shape

        dot_product = cp.dot(P, P.T)
        dot_product_zeros = cp.dot(1 - P, (1 - P).T)
        matches = dot_product + dot_product_zeros
        sim_matrix = matches / L

        # 2. Use a Boolean Mask instead of indices
        # cp.tri() creates a mask of 1s. k=-1 excludes the diagonal.
        # This mask lives entirely on the GPU.
        mask = cp.tri(N, k=-1).T == 1 
        
        # 3. Pull values where mask is True
        upper_tri_values = sim_matrix[mask]
        
        # 4. Calculate mean and explicitly move result to CPU
        mean_sim = cp.mean(upper_tri_values)
        return float(mean_sim.get())

    def compute_jaccard_similarity(self): 
        # 1. Ensure P is definitely a CuPy array
        P = cp.asarray(self.population_predictions)
        N = P.shape[0]

        # 2. Intersection: Dot product gives count of (1,1) matches
        intersection = cp.dot(P, P.T)

        # 3. Row sums: Count of 1s in each individual
        row_sums = P.sum(axis=1)

        # 4. Union: |A| + |B| - |A ∩ B|
        union = row_sums[:, None] + row_sums[None, :] - intersection

        # 5. Jaccard Matrix = Intersection / (Union + epsilon)
        jaccard_matrix = intersection / (union + 1e-8)
        
        # 6. NUCLEAR FIX: Create a Boolean Mask directly on the GPU
        # cp.tri creates a lower triangle; .T makes it upper; k=-1 excludes the diagonal.
        # This creates a matrix of True/False values entirely in GPU memory.
        mask = cp.tri(N, k=-1).T == 1 
        
        # 7. Extract values where mask is True (Direct GPU-native slicing)
        upper_tri_values = jaccard_matrix[mask]
        
        # 8. Calculate mean and explicitly move result to CPU
        mean_jaccard = cp.mean(upper_tri_values)
        
        return float(mean_jaccard.get())



    def run_generation(self, generation):

        def mutate_gene(diff1, diff2, k,F):
            if isinstance(self.parameter_dic[k][0], int) and isinstance(self.parameter_dic[k][1], int): # If the gene we are mutating had integer bounds, than we need to return an integer from the mutation
                mutated_gene =  int(round( (diff1[k] - diff2[k]) * F ) )
            else: 
                mutated_gene = (diff1[k] - diff2[k]) * F 
        
            return mutated_gene

        for index, individual in self.individual_dic.items(): # Loop over all individuals i.e. everyone gets to be a primary parent

            itt = 1
            if index % 10 == 0:
                print(f'Generation: {generation}, individual: {index}')

            while True:

                

                parent_1 = individual # Primary parent
                parent_1_fitness = self.individual_fitness[index]
                
                if parent_1_fitness == self.best_fitness_current_generation and self.ran_elitism == True:
                    self.ran_elitism = False
                    break # If the individual is already the best in the generation we skip mutation (elitism)

                else:

                    while True: # Select a different individual as the second parent, and make sure he isn't the same as the first parent
                        parent_2_index = np.random.randint(0, self.pop_size)
                        if parent_2_index != index:
                            break
                    parent_2 = self.individual_dic[parent_2_index]

                    while True: # Select diff1 and make sure he is different than the first and second parent
                        diff1_index = np.random.randint(0, self.pop_size)
                        if diff1_index != index and diff1_index != parent_2_index:
                            break

                    diff1 = self.individual_dic[diff1_index]

                    while True: # Select diff2 and make sure he is different than the first and second parent
                        diff2_index = np.random.randint(0, self.pop_size)
                        if diff2_index != index and diff2_index != parent_2_index:
                            break
                    diff2 = self.individual_dic[diff2_index]

                    if self.shrinkage_constant is None: # If no shrinkage constant is given we do dithering i.e. F~U(0.4,0.8)
                        F = np.random.uniform(0.4, 0.8)
                    else:
                        F = self.shrinkage_constant

                    mutation_vector = {k: mutate_gene(diff1, diff2, k , F) for k in self.parameter_dic.keys()} # Create the mutation vector by subtracting the two different individuals
                    child = {}

                    for m in range(0, len(mutation_vector.keys())):

                        probability = np.random.rand()

                        if probability >= self.mutation_rate: # If we hit this we keep the value from parent 1
                            key = list(mutation_vector.keys())[m] # Get the key at position m
                            child[key] = parent_1[key] # Keep the value from parent 1
                            
                        else: # Otherwise we mutate
                            key = list(mutation_vector.keys())[m] # Get the key at position m
                            child[key] = np.clip(parent_2[key] + mutation_vector[key], self.parameter_dic[key][0], self.parameter_dic[key][1]) # Add the mutation vector to the parent 1 value
                            # This also ensures that the mutated value stays within the bounds specified by the parameter dictionary

                    child = child | self.fixed_parameter_dic # Add the fixed parameters to the child

                    model_ = self.model(**child) # Intialize the model with the hyper parameters of the individual

                    try:
                        model_.fit(self.X, self.y)

                        # Calculate the fitness of the child
                        if isinstance(self.fitness_function, str) == False: # Calculate the fitness using an inbuilt metric if the fitness function is not a string otherwise call the other type

                            if self.model_type == 'classifier':

                                if self.use_walk_forward == False: # If NOT a walkforward
                                    child_fitness, child_returns = self.calculate_fitness_inbuilt_metric(model_)
                                else: # If a walkforwards
                                    child_fitness, child_returns = self.calculate_fitness_inbuilt_metric_walkforward(child)

                            else:

                                if self.model_type == 'regressor':
                                    if self.use_walk_forward == False:
                                        child_fitness = self.calculate_fitness_inbuilt_metric(model_)
                                    else:
                                        child_fitness = self.calculate_fitness_inbuilt_metric_walkforward(child)

                        else: # This implements the martin ratio fitness function
                            if self.use_walk_forward == False: # Standard
                                child_fitness, child_returns = self.calculate_fitness_martin_ratio(model_)

                            else: # using walk forward
                                child_fitness, child_returns = self.calculate_fitness_martin_ratio_walkforward(child)

                        if child_fitness < parent_1_fitness: # If the child is better than parent 1 we replace parent 1
                            self.individual_dic[index] = child
                            self.individual_fitness[index] = child_fitness
                            self.individual_returns[index] = pd.Series(child_returns).sum() if self.model_type == 'classifier' else None
                            self.population_predictions[index, :] = self.most_recent_oos_predictions

                        #print(f'Ind {index} ran successfully')
                        break
                    
                    except Exception as e:
                        # This here could be useful for verbosity checks
                        print("-" * 60)
                        print(f"Error at Ind {index}, attempt {itt}")
                        traceback.print_exc()  # This prints the full stack trace including line numbers
                        print("-" * 60)
                        itt += 1

    def save_best_params(self, generation): # This function saves the best performing indivdual of the current generation to a json file
        # Find index of the best fitness (minimum because we minimize)
        best_idx = np.argmin(self.individual_fitness)
        best_params = self.individual_dic[best_idx].copy()
        best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best_params.items()}
        best_score = float(self.individual_fitness[best_idx])
        
        # Calculate the Returns based on the best parameters
        

        # Structure the data
        if self.model_type == 'classifier':
            returns = self.individual_returns[best_idx]
            output = {
                "generation": generation,
                'fitness_function': str(self.fitness_function) if not isinstance(self.fitness_function, str) else self.fitness_function,
                "best_fitness": best_score,
                "returns": returns,
                "parameters": best_params,
                "model_type": str(self.model.__name__)
            }
        else: # For a regressor currently we do not support return calculation
            output = {
                "generation": generation,
                'fitness_function': str(self.fitness_function) if not isinstance(self.fitness_function, str) else self.fitness_function,
                "best_fitness": best_score,
                "parameters": best_params,
                "model_type": str(self.model.__name__)
            }
        
        # Save to a file named by generation
        file_path = os.path.join(self.results_dir, f"best_gen_{generation}.json")
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4)
            
        print(f"--> Saved best parameters for Gen {generation} to {file_path}")

    def run_best_individual_optimization(self, similarity_score, sigma_base = 0.1):

        # 1. Shrinkage Factor: Narrows as similarity increases
        # At Jaccard 0.9, we are doing very fine-grained local search
        shrinkage_factor = max(0.01, (1.0 - similarity_score))

        num_nudges = int(1 + (15 * similarity_score)) # The basis of the hill climbing algorithm i.e. how many steps we do

        ind_best_index = np.argmin(self.individual_fitness)
        best_individual = self.individual_dic[ind_best_index]
        best_individual_fitness = self.individual_fitness[ind_best_index]

        # Optimize only the hyperparameters that are numeric
        keys_to_be_optimized = list(key for key,value in self.parameter_dic.items() if isinstance(value, list) and len(value) > 0 and all(isinstance(x, (int, float)) for x in value) )

        key = np.random.choice(keys_to_be_optimized)
        lower, upper = self.parameter_dic[key][0], self.parameter_dic[key][1]

            # Define the param range
        param_range = upper - lower


        for _ in range(num_nudges):

            # Preturb
            nudge = np.random.normal(0, param_range * sigma_base * shrinkage_factor)
            candidate = copy.deepcopy(best_individual)
            candidate[key] = np.clip(candidate[key] + nudge, lower, upper)

            # Integer Handling (Optional)
            # If the parameter must be a whole number (like 'depth' or 'count')
            if isinstance(lower, int) and isinstance(upper, int):
                candidate[key] = int(round(candidate[key]))

            model_ = self.model(**candidate) # Intialize the model with the hyper parameters of the individual
            

            # Evaluate Candidate using the fitness function
            if isinstance(self.fitness_function, str) == False: # Calculate the fitness using an inbuilt metric if the fitness function is not a string otherwise call the other type
            
                if self.model_type == 'classifier':
                    if self.use_walk_forward == False: # If NOT a walkforward
                        model_.fit(self.X, self.y)
                        fitness_, returns = self.calculate_fitness_inbuilt_metric(model_)
                    else: # If a walkforward
                        fitness_, returns = self.calculate_fitness_inbuilt_metric_walkforward(candidate)
                else:
                    if self.use_walk_forward == False:
                        model_.fit(self.X, self.y)
                        fitness_ = self.calculate_fitness_inbuilt_metric(model_)
                    else:
                        fitness_ = self.calculate_fitness_inbuilt_metric_walkforward(candidate)
            
            else:
                if self.use_walk_forward == False:
                    model_.fit(self.X, self.y)
                    fitness_, returns = self.calculate_fitness_martin_ratio(model_)
                else:
                    fitness_, returns = self.calculate_fitness_martin_ratio_walkforward(candidate)

            if fitness_ < best_individual_fitness: # If the candidate is better than the best individual we replace it
                self.individual_dic[ind_best_index] = candidate
                self.individual_fitness[ind_best_index] = fitness_
                self.individual_returns[ind_best_index] = pd.Series(returns).sum() if self.model_type == 'classifier' else None
                #print(f"Optimization --> Improved best individual fitness from {best_individual_fitness} to {fitness_}")
                best_individual_fitness = fitness_

def calculate_1day_premium(iv_annualized, r=0.0):
    T = 1 / 365
    sigma = iv_annualized
    
    # Since S = K, S/K = 1 and ln(1) = 0
    d1 = (r + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # BSM for Call / S = N(d1) - e^(-rT) * N(d2)
    premium_pct = norm.cdf(d1) - np.exp(-r * T) * norm.cdf(d2)
    
    return premium_pct * 0.9

def fetch_data_class(include_only_weekday = True, return_ohlc = True):
    path_to_folder = r"C:\Users\I'm the best\Documents\a\Infrastructure\Feature_Generation"
    if path_to_folder not in sys.path:
        sys.path.append(path_to_folder)
    
    from feature_generator import Data_Store
    from hawkes import hawkes_process
    from reversability import rw_ptsr

    df_btc = pd.read_excel(r"C:\Users\I'm the best\Documents\a\Deribit Downloader\btc_ohlc_1hr.xlsx")
    df_btc['ticks'] = pd.to_datetime(df_btc['ticks'])
    df_btc.set_index('ticks', inplace=True)
    df_dvol = pd.read_excel(r"C:\Users\I'm the best\Documents\a\Deribit Downloader\deribit_vol_index.xlsx")  
    df_dvol['timestamp'] = pd.to_datetime(df_dvol['timestamp'])
    df_dvol.set_index('timestamp', inplace=True)
    df = pd.merge(df_btc, df_dvol, left_index=True, right_index=True, how='inner')


    # This is used to aggregate to daily
    agg_logic = {
        'open_x': 'first',
        'high_x': 'max',
        'low_x': 'min',
        'close_x': 'last',
        'volume': 'sum',
        'open_y': 'first',
        'high_y': 'max',
        'low_y': 'min',
        'close_y': 'last'
    }
    
    df = df.resample('1D', offset='9h').apply(agg_logic)
    df['abs_daily_return'] = np.abs( (df['close_x'].shift(-1) - df['close_x'])/df['close_x'] )
    df['premium'] = df['close_y'].apply(lambda x: 2 * calculate_1day_premium(x/100) )
    df['weekday'] = df.index.weekday < 5
    df['y'] = df['premium'] > df['abs_daily_return']
    print(df)

    df.rename(columns={'open_x': 'open', 'high_x': 'high', 'low_x': 'low', 'close_x': 'close'}, inplace=True)

    df['ATR'] = ta.atr(high=np.log(df['high']), low=np.log(df['low']), close=np.log(df['close']), length=30)
    df['RSI'] = ta.rsi(close=df['close'], length=30)
    norm_range = ( np.log(df['high']) - np.log(df['low']) ) / df['ATR']
    df['Hawkes'] = hawkes_process(norm_range, 0.1)
    df['Reversability'] = rw_ptsr(df['close'], 30)

    if include_only_weekday:
        df = df[df['weekday'] == True]

    df = df.dropna()
    X = df[['ATR', 'RSI', 'Hawkes', 'Reversability', 'premium', 'weekday']].dropna()
    y = df[['y']].loc[X.index]

    if return_ohlc:
        return X, y, df[['open', 'high', 'low', 'close', 'abs_daily_return', 'premium']]

    return X, y


def fetch_data_regression(include_only_weekday = True, return_ohlc = True):
    path_to_folder = r"C:\Users\I'm the best\Documents\a\Infrastructure\Feature_Generation"
    if path_to_folder not in sys.path:
        sys.path.append(path_to_folder)
    
    from feature_generator import Data_Store
    from hawkes import hawkes_process
    from reversability import rw_ptsr

    df_btc = pd.read_excel(r"C:\Users\I'm the best\Documents\a\Deribit Downloader\btc_ohlc_1hr.xlsx")
    df_btc['ticks'] = pd.to_datetime(df_btc['ticks'])
    df_btc.set_index('ticks', inplace=True)
    df_dvol = pd.read_excel(r"C:\Users\I'm the best\Documents\a\Deribit Downloader\deribit_vol_index.xlsx")  
    df_dvol['timestamp'] = pd.to_datetime(df_dvol['timestamp'])
    df_dvol.set_index('timestamp', inplace=True)
    df = pd.merge(df_btc, df_dvol, left_index=True, right_index=True, how='inner')


    # This is used to aggregate to daily
    agg_logic = {
        'open_x': 'first',
        'high_x': 'max',
        'low_x': 'min',
        'close_x': 'last',
        'volume': 'sum',
        'open_y': 'first',
        'high_y': 'max',
        'low_y': 'min',
        'close_y': 'last'
    }
    
    df = df.resample('1D', offset='9h').apply(agg_logic)
    df['abs_daily_return'] = np.abs((df['close_x'].shift(-1) - df['close_x'])/df['close_x'])
    df['premium'] = df['close_y'].apply(lambda x: 2 * calculate_1day_premium(x/100) )
    df['weekday'] = df.index.weekday < 5
    df['y'] = df['premium'] - df['abs_daily_return']


    df.rename(columns={'open_x': 'open', 'high_x': 'high', 'low_x': 'low', 'close_x': 'close'}, inplace=True)

    df['ATR'] = ta.atr(high=np.log(df['high']), low=np.log(df['low']), close=np.log(df['close']), length=30)
    df['RSI'] = ta.rsi(close=df['close'], length=30)
    norm_range = ( np.log(df['high']) - np.log(df['low']) ) / df['ATR']
    df['Hawkes'] = hawkes_process(norm_range, 0.1)
    df['Reversability'] = rw_ptsr(df['close'], 30)

    if include_only_weekday:
        df = df[df['weekday'] == True]

    df = df.dropna()
    X = df[['ATR', 'RSI', 'Hawkes', 'Reversability', 'premium', 'weekday']].dropna()
    y = df[['y']].loc[X.index]

    if return_ohlc:
        return X, y, df[['open', 'high', 'low', 'close', 'abs_daily_return', 'premium']]

    return X, y




if __name__ == "__main__":

    # Don't forge to set tree_method as 'hist' and device as 'cuda' or 'gpu' if so you can play around with sampling_method as 'gradient_based' or 'uniform'
    # Also play around with 'grow_policy' as 'depthwise' or 'lossguide' for better results
    '''
    Also an "objective" function needs to be defined:
        For regressions: reg:squarederror, reg:squaredlogerror, reg:logistic, reg:pseudohubererror, reg:absoluteerror
        For classifications: binary:logistic (produces probability), binary:hinge (produces class), multi:softprob(multiclass for softmax needs also num_classes, gives label),

    For 'eval_metric':
        For regressions: 'rmse', 'mae', 'logloss'
        For classification: 'error' (accuracy), error@t (same as error but you sumpliment t with a threshold value for classifying), 'auc', 'aucpr' /Both those need binary:logistic objective or multi:softprob/
        'merror' (for multiclass classification), 'map' (mean average precision), 'pre' (Precision for only top k classes) /'map' and 'pre' can add map@n where n is the top n classes to consider/

    '''
    
    X, y, ohlc = fetch_data_class()
    #X, y, ohlc = fetch_data_regression()



    fixed_parameter_dic_xg_boost_class = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'aucpr',#aucpr
        'device': 'cuda'
    }

    fixed_parameter_dic_xg_boost_regression = {
        'objective': 'reg:absoluteerror',
        'tree_method': 'hist',
        'eval_metric': 'mae',
        'device': 'cuda'
    }

    parameter_dic_xg_boost = {
        'eta': [0.001, 1.0],
        'n_estimators': [10, 1000],
        'max_depth': [1, 30],
        'subsample': [0.5, 1.0],
        'gamma': [1e-16, 10.0],
        #'max_delta_step': [1e-16, 10], # Useful in Classification if the classes are very imbalanced /Always initialize with 0 as well/
        'min_child_weight': [1, 20] # 1 is default, 2-10 moderate regularization, 10+ strong regularization
    }

    de = Differential_Evolution(X = X, y= y, fitness_function= 'Martin Ratio', pop_size=10, overint=1,
                                mutation_rate= 0.5,
                                model = XGBClassifier,
                                parameter_dic = parameter_dic_xg_boost, 
                                fixed_parameter_dic = fixed_parameter_dic_xg_boost_class,
                                ohlc= ohlc, use_walk_forward= False, max_generations= 70)
