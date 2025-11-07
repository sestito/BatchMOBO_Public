


def CheckStoppingConditions(X, Y, conditions: list = []):
    to_stop = False
    for condition in conditions:
        if condition(X, Y):
            to_stop = True
            break
    
    return to_stop