import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
#͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
# pgmpy.sampling.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
# pgmpy.factors.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
# pgmpy.estimators.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆    
    raise NotImplementedError
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    # TODO: set the probability distribution for each node͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError    
    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
    return double0_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError    
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆    
    raise NotImplementedError
    return posterior # list 


def getA(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "A" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    a = []
    normalizer = 0
    for i in range(4):
        normalizer += (team_table[i] * match_table[evidence[3]][i][evidence[1]] * 
                       match_table[evidence[5]][i][evidence[2]])
    for i in range(4):
        unnorm_prob = (team_table[i] * match_table[evidence[3]][i][evidence[1]] * 
                       match_table[evidence[5]][i][evidence[2]])
        a.append(unnorm_prob)
    return numpy.array(a)/normalizer


def getBvC(bayes_net, B, C):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of match outcomes for "BvC" given the skill levels of B and C as evidence
    Parameter: 
    : bayes net: Baysian Model Object
    : B: int representing team B's skill level
    : C: int representing team C's skill level
    """
    BvC_cpd = bayes_net.get_cpds('BvC')
    match_table = BvC_cpd.values
    bvc = []
    for i in range(0, 3):
        bvc.append(match_table[i][B][C])
    return bvc   


def getB(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "B" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    B_cpd = bayes_net.get_cpds("B")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = B_cpd.values
    b = []
    normalizer = 0
    for i in range(4):
        normalizer += (team_table[i] * match_table[evidence[3]][evidence[0]][i] * 
                       match_table[evidence[4]][i][evidence[2]])
    for i in range(4):
        unnorm_prob = (team_table[i] * match_table[evidence[3]][evidence[0]][i] * 
                       match_table[evidence[4]][i][evidence[2]])
        b.append(unnorm_prob)
    return numpy.array(b)/normalizer


def getC(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "C" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    C_cpd = bayes_net.get_cpds("C")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = C_cpd.values
    c = []
    normalizer = 0
    for i in range(4):
        normalizer += (team_table[i] * match_table[evidence[5]][i][evidence[0]] * 
                       match_table[evidence[4]][evidence[1]][i])
    for i in range(4):
        unnorm_prob = (team_table[i] * match_table[evidence[5]][i][evidence[0]] * 
                       match_table[evidence[4]][evidence[1]][i])
        c.append(unnorm_prob)
    return numpy.array(c)/normalizer

def calculateMH(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns the probability of a state.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    AvB_cpd = bayes_net.get_cpds('AvB').values
    BvC_cpd = bayes_net.get_cpds('BvC').values
    CvA_cpd = bayes_net.get_cpds('CvA').values
    skill_dist = [0.15, 0.45, 0.30, 0.10]
    A_skill_prob = skill_dist[evidence[0]]
    B_skill_prob = skill_dist[evidence[1]]
    C_skill_prob = skill_dist[evidence[2]]
    AvB_outcome_prob = AvB_cpd[evidence[3]][evidence[0]][evidence[1]]
    BvC_outcome_prob = BvC_cpd[evidence[4]][evidence[1]][evidence[2]]
    CvA_outcome_prob = CvA_cpd[evidence[5]][evidence[2]][evidence[0]]
    
    
    return (A_skill_prob * B_skill_prob * C_skill_prob * AvB_outcome_prob * 
            BvC_outcome_prob * CvA_outcome_prob)


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
