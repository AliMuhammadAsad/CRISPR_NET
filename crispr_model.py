from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# Define the structure of the Bayesian Network with 31 nodes
edges = [
    ('Node1', 'Node31'), ('Node2', 'Node31'), ('Node3', 'Node31'), ('Node4', 'Node31'),
    ('Node5', 'Node31'), ('Node6', 'Node31'), ('Node7', 'Node31'), ('Node8', 'Node31'),
    ('Node9', 'Node31'), ('Node10', 'Node31')
]
nodes = 11

"""('Node11', 'Node31'), ('Node12', 'Node31'),
    ('Node13', 'Node31'), ('Node14', 'Node31'), ('Node15', 'Node31'), ('Node16', 'Node31'),
    ('Node17', 'Node31'), ('Node18', 'Node31'), ('Node19', 'Node31'), ('Node20', 'Node31'),
    ('Node21', 'Node31'), ('Node22', 'Node31'), ('Node23', 'Node31'), ('Node24', 'Node31'),
    ('Node25', 'Node31'), ('Node26', 'Node31'), ('Node27', 'Node31'), ('Node28', 'Node31'),
    ('Node29', 'Node31'), ('Node30', 'Node31') """
# Initialize the Bayesian Network
model = BayesianNetwork(edges)
# print(model)
model_graph = model.to_graphviz()

model_graph.draw("ss.png", prog = "dot")

# Define CPDs (Conditional Probability Distributions) for each node
# For simplicity, let's assume binary nodes with random probabilities
for i in range(1, nodes):
    # Nodes 1 to 30 have variable_card=4 with equal probability distribution
    cpd = TabularCPD(variable=f'Node{i}', variable_card=4, values=[[0.25], [0.25], [0.25], [0.25]])
    model.add_cpds(cpd)

k = 4**10
# val = 1/k
# Define CPD for Node31 with variable_card=2 with equal probability
cpd_31 = TabularCPD(variable='Node31', variable_card=2, values=[[0.5]*k, [0.5]*k], 
                    evidence=[f'Node{i}' for i in range(1, nodes)], evidence_card=[4]*(nodes-1))
model.add_cpds(cpd_31)

# Check if the model is valid
assert model.check_model(), "The model is not valid!"

# Perform inference
inference = VariableElimination(model)

# Example query
query_result = inference.query(variables=['Node31'], evidence={'Node1': 1})
print(query_result)
