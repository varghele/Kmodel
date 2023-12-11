class KModel:
    def __init__(self, df):
        self.df = df
        
    def remove_outliers(self, clmns):
        """
        Removing productivity data that has metrics of 0 or >1
        :param clmns: list of column names
        :return: Dataframe with removed outliers
        """
        # Failsafe to correctly slice the DF if somebody enters only one outlier criteria
        if len(clmns)==1:
            # Remove "errors", when productivity was 0
            self.df = self.df[self.df[clmns[0]] != 0]
            # Remove "errors", where productivity is unreasonably high
            self.df = self.df[self.df[clmns[0]] < 1]
        else:
            for clmn in clmns:
                # Remove "errors", when productivity was 0
                self.df = self.df[self.df[clmn] != 0]
                # Remove "errors", where productivity is unreasonably high
                self.df = self.df[self.df[clmn] < 1]
        
        return self.df
    
    
    def weight_by_date(self, daterange=90):
        """
        Assign weights to datapoints (older are worth less)
        :param daterange: int that represents how many points(days) of data are considered
        :return: Dataframe with "weight" column
        """
        # Get unique dates
        unq_dates = self.df["prod_day"].unique()
        
        # Create weight column
        self.df["weight"] = ""
        
        # Create weight column that creates weight depending on daterange
        weights = [0 for i in range(len(unq_dates)-daterange)] + [i for i in range(1,daterange+1)][::-1]
        for i in range(len(unq_dates)):
            self.df["weight"].iloc[self.df.index[self.df['prod_day'] == unq_dates[::-1][i]].tolist()] = weights[i]
            
        # Eliminate values with 0 weight
        self.df = self.df[self.df["weight"] != 0]
        
        return self.df
    
    
    # Fairly fast for many datapoints, less fast for many costs, somewhat readable
    def is_pareto_efficient_simple(self, values, n_fronts=3):
        """
        Find the pareto-efficient points
        :param values: An (n_points, n_values) array
        :param n_fronts: int that defines how many pareto fronts are considered
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        pareto_efficient_idx = np.zeros(values.shape[0], dtype = bool)
        vmean = np.mean(values,axis=0)
        # Consider multiple pareto fronts
        for n in range(n_fronts):
            is_efficient = np.ones(values.shape[0], dtype = bool)
            for i, c in enumerate(values):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(values[is_efficient]>=c, axis=1)  # Keep any point with a higher effc
                    is_efficient[i] = True  # And keep self
            
            # Append indices if they are above mean
            above_mean = np.all(values>vmean,axis=-1)
            is_efficient = np.all(np.stack([is_efficient, above_mean]),axis=0)
            pareto_efficient_idx = np.where(is_efficient,True,pareto_efficient_idx)
            
            # "Remove" the datapoints from consideration
            values = np.array([values[v]-1 if is_efficient[v]==True else values[v] for v in range(values.shape[0])])
            
        return pareto_efficient_idx
    
    
    def get_best_values(self, clmns, n_fronts=3):
        """
        :param clmns: list of column names
        :param n_fronts: int that defines how many pareto fronts are considered
        :return: Dataframe of only the "best" values
        """
        # If "best" is determined by only one parameter, simple pick values to the right of the mean
        if len(clmns)==1:
            mean = np.mean(self.df[clmns[0]])
            self.df = self.df[self.df[clmns[0]]>mean]
        else:
            #Get pareto optimal solutions
            par = self.is_pareto_efficient_simple(np.array(self.df[clmns].values.tolist()), n_fronts)
            # Select pareto optimal points from dataframe
            self.df = self.df[par]
            
        return self.df
    
    
    def get_aggregate(self,clmns,target_id_clmn_name,
                      target_clmn_name):
        """
        :param clmns: list of column names
        :param target_id_clmn_name: string of name of ID column
        :param target_clmn_name: string of name of target column
        :return:
        """
        results = []
        # Only select columns that are not targets or weight
        out_cols = [i for i in self.df.columns if (i not in clmns) and (i!="weight") and (i!=target_clmn_name)]
        for i in np.unique(self.df[target_id_clmn_name].values):
            # Get aggregated centerlining value
            cl_value = np.sum((self.df[self.df[target_id_clmn_name]==i][target_clmn_name]*self.df[self.df[target_id_clmn_name]==i]["weight"]))/np.sum(self.df[self.df[target_id_clmn_name]==i]["weight"])
            # Only select columns that are not targets or weight
            results.append(self.df[self.df[target_id_clmn_name]==i][out_cols].iloc[0].values.tolist()+[cl_value])
        self.df = pd.DataFrame(results, columns = out_cols+[target_clmn_name])
            
        return self.df
    
        
    def forward(self, clmns, target_id_clmn_name, target_clmn_name, daterange=90, n_fronts=3):
        """
        :param clmns:
        :param target_id_clmn_name: string of name of ID column
        :param target_clmn_name: string of name of target column
        :param daterange: int that represents how many points(days) of data are considered
        :param n_fronts: int that defines how many pareto fronts are considered        
        :return:
        """
        # Remove outlier data (too far removed from mean)
        self.remove_outliers(clmns=clmns)
        
        # Apply weight column that weighs data by date
        self.weight_by_date(daterange=daterange)
        
        # only select values that are better than the mean (or on the pareto front)
        self.get_best_values(clmns=clmns, n_fronts=n_fronts)
        
        # average between the values
        self.get_aggregate(clmns=clmns, target_id_clmn_name=target_id_clmn_name,
                           target_clmn_name=target_clmn_name)
        
        # Return results
        return self.df
        
        
