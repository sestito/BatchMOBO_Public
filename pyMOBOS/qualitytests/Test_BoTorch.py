class Test_BoTorch:
    def __init__(self, aquisition_function, name = "", batch_size = 1):
        self.aquisition_function = aquisition_function
        self.name = name
        self.batch_size = batch_size
        
    
    def __call__(self, X, Y, X_Bounds):
        ac = self.aquisition_function(X, Y, X_Bounds)
        new_samples = ac(self.batch_size)
        return new_samples
