class Network:
    def __init__(self, layers):
        self.layers=layers
    def run(self, input):
        vect=input
        for layer in self.layers:
            vect=layer.run(vect)
        return vect
    def train(self, dataset, targets):
        pass
            

