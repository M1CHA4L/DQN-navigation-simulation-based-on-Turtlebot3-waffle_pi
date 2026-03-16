import graphviz

def generate_dqn_diagram():
    dot = graphviz.Digraph(comment='DQN Architecture', format='png')
    dot.attr(rankdir='TB', splines='ortho')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='12', margin='0.2')
    
    # Define Nodes
    dot.node('Input', 'State Input\n(22 dim: Laser, Distance, Angle)', fillcolor='#E1D5E7')  # Purple-ish
    
    with dot.subgraph(name='cluster_hidden') as c:
        c.attr(style='dashed,rounded', label='Hidden Layers', color='gray', fontname='Arial')
        c.node('L1', 'Linear (256) \n + ReLU Activation', fillcolor='#D5E8D4')  # Green-ish
        c.node('L2', 'Linear (256) \n + ReLU Activation', fillcolor='#D5E8D4')
        c.node('L3', 'Linear (128) \n + ReLU Activation', fillcolor='#D5E8D4')
    
    dot.node('Output', 'Action Q-Values Output\n(5 discrete actions)', fillcolor='#FFE6CC', shape='ellipse')  # Orange-ish

    # Connect Nodes
    dot.edge('Input', 'L1')
    dot.edge('L1', 'L2')
    dot.edge('L2', 'L3')
    dot.edge('L3', 'Output')

    # Render
    dot.render('DQN_Block_Architecture', cleanup=True)
    print("Created DQN_Block_Architecture.png")


def generate_ppo_diagram():
    dot = graphviz.Digraph(comment='PPO Architecture', format='png')
    dot.attr(rankdir='TB', splines='ortho')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='12', margin='0.2')

    # Input Node
    dot.node('Input', 'State Input\n(22 dim: Laser, Distance, Angle)', fillcolor='#E1D5E7')

    # Shared Network subgraph
    with dot.subgraph(name='cluster_shared') as c:
        c.attr(style='dashed,rounded', label='Shared Feature Extractor', color='gray', fontname='Arial')
        c.node('S1', 'Linear (256) \n + Tanh Activation', fillcolor='#E1D5E7')
        c.node('S2', 'Linear (256) \n + Tanh Activation', fillcolor='#E1D5E7')
        c.edge('S1', 'S2')

    # Dummy node to help routing
    dot.node('Split', '', shape='point', width='0', height='0')

    # Actor Head subgraph
    with dot.subgraph(name='cluster_actor') as c:
        c.attr(style='dashed,rounded', label='Actor Head (Policy)', color='#1b78c4', fontname='Arial', fontcolor='#1b78c4')
        c.node('A1', 'Linear (128) \n + Tanh Activation', fillcolor='#DAE8FC')
        c.node('A2', 'Linear (2) \n + Tanh Activation', fillcolor='#DAE8FC')
        c.node('ActorOut', 'Action Mean\n(Continuous: Linear, Angular)', fillcolor='#FFE6CC', shape='ellipse')
        c.edge('A1', 'A2')
        c.edge('A2', 'ActorOut')

    # Critic Head subgraph
    with dot.subgraph(name='cluster_critic') as c:
        c.attr(style='dashed,rounded', label='Critic Head (Value)', color='#b85450', fontname='Arial', fontcolor='#b85450')
        c.node('C1', 'Linear (128) \n + Tanh Activation', fillcolor='#F8CECC')
        c.node('C2', 'Linear (1)', fillcolor='#F8CECC')
        c.node('CriticOut', 'State Value\n(V(s))', fillcolor='#FFF2CC', shape='ellipse')
        c.edge('C1', 'C2')
        c.edge('C2', 'CriticOut')

    # Connections
    dot.edge('Input', 'S1')
    dot.edge('S2', 'Split', dir='none')
    dot.edge('Split', 'A1')
    dot.edge('Split', 'C1')

    # Render
    dot.render('PPO_Block_Architecture', cleanup=True)
    print("Created PPO_Block_Architecture.png")

if __name__ == '__main__':
    print("Generating Academic Block Diagrams...")
    generate_dqn_diagram()
    generate_ppo_diagram()
    print("Done!")