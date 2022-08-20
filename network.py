import data
import numpy as np

class Network:
    def __init__(self, layers, loss_function):
        self.layers=layers
        self.loss_function=loss_function
    def run(self, input):
        vect=input
        for layer in self.layers:
            vect=layer.run(vect)
        return vect
    
    def get_network_overview(self, input): 
        vect=input
        nodes=[vect]
        for layer in self.layers:
            vect=layer.run(vect)
            nodes.append(vect)
        return nodes
    
    def average_loss_and_nodes(self, miniset, minitargets):
        average_loss=np.zeros(self.layers[-1].dim_output)
        average_nodes=None
        for i, input in enumerate(miniset):
            nodes=self.get_network_overview(input)
            loss=self.loss_function(nodes[-1], minitargets[i])
            if average_nodes is None:
                average_nodes=nodes
            else: 
                average_nodes+=nodes
            average_loss+=loss
        return average_loss/len(miniset), average_nodes/len(miniset)
    
    def backpropagation(self,average_loss, average_nodes):
        error_next_layer = average_loss
        for i in range(len(self.layers)-1, -1, -1):
            error_next_layer=self.layers[i].learn(average_nodes[i], error_next_layer)
    
    def train(self, dataset, targets, lenghtminiset):
        minisets, minitargets=data.get_minisets(dataset, targets, lenghtminiset)
        for i in range(len(minisets)):
            average_loss, average_nodes=self.average_loss_and_nodes(minisets[i], minitargets[i])
            self.backpropagation(average_loss, average_nodes)
    


    
    

            



            

