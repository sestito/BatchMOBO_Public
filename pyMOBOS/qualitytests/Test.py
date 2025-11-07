


class Test:
    def __init__(self, surrogate, aquisition_function, name = "", batch_size = 1, quality_metrics = None):
        self.surrogate = surrogate
        self.aquisition_function = aquisition_function
        self.name = name
        self.batch_size = batch_size
        self.quality_metrics = quality_metrics
    
    def __call__(self, X, Y, X_Bounds):
        surrogate = self.surrogate(X, Y)

        if self.quality_metrics == None:
            ac = self.aquisition_function(X, Y, X_Bounds, surrogate)
        else:
            ac = self.aquisition_function(X, Y, X_Bounds, surrogate, self.quality_metrics)
        
        if self.batch_size == 1:
            new_samples = ac()
        
        else:
            new_samples = ac(self.batch_size)
        
        return new_samples
