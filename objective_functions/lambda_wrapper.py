# Wraps objectives that are specified using lambda function or plain function into a wrapper that has get_name() properly implemented
class lambda_wrapper:
    def __init__(self, objective_implementation, objective_name):
        self.objective_implementation = objective_implementation
        self.objective_name = objective_name
    
    def __call__(self, recommendation_list, context, m=None):
        return self.objective_implementation(recommendation_list, context, m)

    def get_name(self):
        return self.objective_name