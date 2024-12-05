import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the model
model = BayesianNetwork([('A', 'B'), ('B', 'C')])

# Define the CPDs
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.8], [0.2]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.7, 0.1], [0.3, 0.9]], evidence=['A'], evidence_card=[2])
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=['B'], evidence_card=[2])

# Add the CPDs to the model
model.add_cpds(cpd_a, cpd_b, cpd_c)
model.check_model()

# Inference
inference = VariableElimination(model)

# Streamlit UI
st.title("PGM Genie")
a_value = st.selectbox("Select value for A", [0, 1])

if st.button("Query C"):
    result = inference.query(variables=['C'], evidence={'A': a_value})
    st.write(f"Probability of C given A={a_value}: {result.values}")