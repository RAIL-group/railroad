import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sctp.param import RobotType
from scipy.stats import gaussian_kde

def plot_plan_exec(graph, plt, name="Graph", gpath=[], dpaths = [], graph_plot=None,
               start_coord=None, goal_coord=None, seed=None, cost=0.0, verbose=False):
    """Plot graph using matplotlib."""
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    if graph_plot is not None:
        ax[0].scatter(start_coord[0], start_coord[1], marker='o', color='r')
        ax[0].text(start_coord[0]-1.0, start_coord[1],'Start',color='blue', fontsize=8)
        ax[0].scatter(goal_coord[0], goal_coord[1], marker='x', color='r')
        ax[0].text(goal_coord[0]+0.2, goal_coord[1],'Goal',color='r', fontsize=8)
        box= plot_sctpgraph(graph_plot, ax[0], verbose=verbose)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xlim(box[0][0]-1.2, box[1][0]+1.2)
        ax[0].set_ylim(box[0][1]-0.5, box[1][1]+1.0)
        ax[0].set_title(f'Seed: {seed} | Initial Graph')
        
    ax[1].scatter(start_coord[0], start_coord[1], marker='o', color='r')
    ax[1].text(start_coord[0]-1.0, start_coord[1],'Start',color='blue', fontsize=8)
    ax[1].scatter(goal_coord[0], goal_coord[1], marker='x', color='r')
    ax[1].text(goal_coord[0]+0.2, goal_coord[1],'Goal',color='r', fontsize=8)
    box = plot_sctpgraph(graph, ax[1], verbose=verbose)
    if len(gpath[0]) > 1: 
        g_colors = ['orange', 'green']
        ax[1].scatter(gpath[0], gpath[1], marker='P', s=4.5, alpha=1.0)
        plot_path_fromPoints(ax=ax[1], xy=gpath, colors=g_colors)
    if dpaths != [] and len(dpaths[0][0]) >1:
        for path in dpaths:
            d_colors = ['blue', 'purple']
            ax[1].scatter(path[0],path[1], marker='s', s=4.5)
            plot_path_fromPoints(ax=ax[1], xy=path, colors=d_colors)
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlim(box[0][0]-1.2, box[1][0]+1.2)
    ax[1].set_ylim(box[0][1]-0.5, box[1][1]+1.0)
    ax[1].set_title(f'Seed: {seed} | Planner: {name} | Cost: {cost:.2f}')
    # plt.title(f'Seed: {seed} | Planner: {name} | Cost: {cost:.2f}')
    
def plot_policy(graph, name="Policy", actions=[], 
               startID=None, goalID=None, seed=None, verbose=False):
    fig, ax = plt.subplots()
    count = 0
    for node in graph.vertices+graph.pois:
        if startID is not None:
            if node.id == startID:
                count += 1
                ax.text(node.coord[0]-1.0, node.coord[1], "Start", color='blue', fontsize=8)
        if goalID is not None:
            if node.id == goalID:
                count += 1
                ax.text(node.coord[0] + 0.3, node.coord[1], "Goal", color='red', fontsize=8) 

    box = plot_sctpgraph(graph, ax, verbose=verbose)
    g_cost = 0.0
    x_drone = []
    if actions != []:
        d_colors = ['yellow', 'blue']
        g_colors = ['orange', 'green']
        g_cost, x_drone = plot_path_fromActions(ax, graph=graph, actions=actions, dcolors=d_colors, gcolors=g_colors)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(box[0][0]-1.0, box[1][0]+1.0)
    ax.set_ylim(box[0][1]-0.5, box[1][1]+0.5)

    plt.title(name+f' | seed = {seed} | cost = {g_cost:.2f}')
    if x_drone == []:
        planner = 'base'
    else:
        planner = 'sctp'
    plt.savefig(f'/data/sctp/sctp_eval_policy_{planner}_seed_{seed}.png')
    plt.show()

def plot_firstAction(graph, action, name="First Action", 
               startID=None, goalID=None, seed=None, verbose=False):
    fig, ax = plt.subplots()
    count = 0
    for node in graph.vertices+graph.pois:
        if startID is not None:
            if node.id == startID:
                count += 1
                ax.text(node.coord[0]-1.0, node.coord[1], "Start", color='blue', fontsize=8)
        if goalID is not None:
            if node.id == goalID:
                count += 1
                ax.text(node.coord[0] + 0.3, node.coord[1], "Goal", color='red', fontsize=8) 

    box = plot_sctpgraph(graph, ax, verbose=verbose)
    g_cost = 0.0
    x_drone = []
    # if actions != []:
    d_colors = ['yellow', 'purple']
    g_colors = ['orange', 'green']
    g_cost, x_drone = plot_path_fromActions(ax, graph=graph, actions=[action], dcolors=d_colors, gcolors=g_colors)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(box[0][0]-1.0, box[1][0]+1.0)
    ax.set_ylim(box[0][1]-0.5, box[1][1]+0.5)

    plt.title(name+f' | seed = {seed} | cost = {g_cost:.2f}')
    if x_drone == []:
        planner = 'base'
    else:
        planner = 'sctp'
    plt.savefig(f'/data/sctp/sctp_eval_policy_{planner}_seed_{seed}.png')
    plt.show()

    
def plot_path_fromPoints(ax, xy, colors):
    dist = 0.0
    rev = 0.2
    x = xy[0]
    y = xy[1]
    for i in range(len(x)-1):
        dist += np.linalg.norm(np.array([x[i],y[i]]) - np.array(np.array([x[i+1],y[i+1]])))
    plot_lines_varyWidthColor(ax, [x, y], dist, rev, colors)

def plot_path_fromActions(ax, graph, actions, dcolors, gcolors):
    g_cost = 0.0
    d_dist = 0.0
    rev = 0.2
    x_robot = []
    y_robot = []
    x_drone = []
    y_drone = []
    last_robot_action = None
    last_drone_action = None
    for a in actions:
        if a.rtype == RobotType.Ground:
            if x_robot != []:
                g_cost += np.linalg.norm(np.array([x_robot[-1],y_robot[-1]]) - np.array(a.start_pose))
            x_robot.append(a.start_pose[0])
            y_robot.append(a.start_pose[1])
            last_robot_action = a
        elif a.rtype == RobotType.Drone:
            if x_drone != []:
                d_dist += np.linalg.norm(np.array([x_drone[-1],y_drone[-1]]) - np.array(a.start_pose))
            x_drone.append(a.start_pose[0])
            y_drone.append(a.start_pose[1])
            last_drone_action = a
    last_vertex = [vertex for vertex in graph.vertices+graph.pois if vertex.id == last_robot_action.target][0]
    g_cost += np.linalg.norm(np.array([x_robot[-1],y_robot[-1]]) - np.array(last_vertex.coord))
    x_robot.append(last_vertex.coord[0])
    y_robot.append(last_vertex.coord[1])
    plot_lines_varyWidthColor(ax, [x_robot, y_robot], g_cost, rev, gcolors)
    if x_drone != []:
        last_vertex = [vertex for vertex in graph.vertices+graph.pois if vertex.id == last_drone_action.target][0]
        d_dist += np.linalg.norm(np.array([x_drone[-1],y_drone[-1]]) - np.array(last_vertex.coord))
        x_drone.append(last_vertex.coord[0])
        y_drone.append(last_vertex.coord[1])
        plot_lines_varyWidthColor(ax, [x_drone, y_drone], d_dist, rev, dcolors)
    return g_cost, x_drone



def plot_sctpgraph(graph, plt, textsize=7, verbose=False):
    x_max = max(enumerate(graph.vertices), key=lambda v: v[1].coord[0])[1].coord[0]
    x_min = min(enumerate(graph.vertices), key=lambda v: v[1].coord[0])[1].coord[0]
    y_max = max(enumerate(graph.vertices), key=lambda v: v[1].coord[1])[1].coord[1]
    y_min = min(enumerate(graph.vertices), key=lambda v: v[1].coord[1])[1].coord[1]
    
    plt.set_ylim(y_min-0.5, y_max+1.0)
    plt.set_xlim(x_min-1.2, x_max+1.0)
    # Plot edges
    count = 0
    for edge in graph.edges:
        x_values = [edge.v1.coord[0], edge.v2.coord[0]]
        y_values = [edge.v1.coord[1], edge.v2.coord[1]]
        
        # print(f"Edge {count}: {edge.v1.id}-{edge.v2.id}")
        count += 1
        plt.plot(x_values, y_values, 'b-', linewidth=1.0, alpha=1.0)
        # Display block probability
        if verbose:
            mid_x = (edge.v1.coord[0] + edge.v2.coord[0]) / 2
            mid_y = (edge.v1.coord[1] + edge.v2.coord[1]) / 2
            costs = f"{edge.cost:.2f}"
            plt.text(mid_x, mid_y+0.25, costs, color='red', fontsize=textsize)

    # Plot nodes
    for node in graph.vertices:
        plt.scatter(node.coord[0], node.coord[1], color='green', s=25)
        if verbose:
            plt.text(node.coord[0], node.coord[1] + 0.2, f"V{node.id}", color='blue', fontsize=textsize)

    for poi in graph.pois:
        plt.scatter(poi.coord[0], poi.coord[1], color='red', s=25)
        if poi.block_status == 1:
            plt.scatter(poi.coord[0], poi.coord[1], color='black', s=8)
        else:
            plt.scatter(poi.coord[0], poi.coord[1], color='white', s=8)
        if verbose:
            plt.text(poi.coord[0]-0.3, poi.coord[1] + 0.25, f"P{poi.id}"+f"/{poi.block_prob:.2f}", color='blue', fontsize=textsize)
    return [[x_min, y_min], [x_max, y_max]]
        

def make_scatter_plot_with_box(data_x, data_y, max_val=None, xlabel='Baseline', ylabel='SCTP'):
    if max_val is None:
        max_val = 1.1 * max(max(data_x), max(data_y))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    gs = fig.add_gridspec(nrows=8, ncols=8)
    f_ax_mid = fig.add_subplot(gs[:, :])
    f_ax_bot = fig.add_subplot(gs[-1, :])
    f_ax_lft = fig.add_subplot(gs[:, 0])

    make_scatter_plot(f_ax_mid, data_x, data_y, max_val)

    f_ax_lft.boxplot(list(data_y), vert=True, showmeans=True)
    f_ax_lft.set_ylim([0, max_val])
    f_ax_lft.set_axis_off()

    f_ax_bot.boxplot(list(data_x), vert=False, showmeans=True)
    f_ax_bot.set_xlim([0, max_val])
    f_ax_bot.set_axis_off()

    f_ax_mid.set_xlabel(xlabel)
    f_ax_mid.set_ylabel(ylabel)

    baseline_cost = np.average(data_x)
    learned_cost = np.average(data_y)
    improv = (baseline_cost - learned_cost) / baseline_cost * 100

    f_ax_mid.set_title(f"{xlabel}: {baseline_cost:.1f}, {ylabel}: {learned_cost:.1f}, Improv: {improv:.1f} %")

    return f_ax_mid

def make_scatter_plot(ax, cost_x, cost_y, max_val):
    y_axis = cost_y
    x_axis = cost_x
    # Calculate the point density
    xy = np.vstack([x_axis, y_axis])
    z = gaussian_kde(xy)(xy)
    colors = plt.get_cmap("Blues")((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.50)

    ax.scatter(x_axis, y_axis, c=colors)
    # Draw a center line
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=0.5, alpha=0.2)

    # Set X-axis and Y-axis up to same range; Using ax.axis('square') gives scaling issues for box plot
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)


def plot_lines_varyWidthColor(ax, xy, total_dist, rev=0.2, color_pair=['orange', 'green']):
    n_points = int(total_dist/rev)
    counter = 0
    # Define color gradient (red to blue)
    colors = np.linspace(0, 1, n_points)
    cmap = LinearSegmentedColormap.from_list('custom', color_pair)
    
    for i in range(len(xy[0])-1):
        dist = np.linalg.norm(np.array([xy[0][i],xy[1][i]]) - np.array([xy[0][i+1],xy[1][i+1]]))
        seg_points = int(dist/rev)
        x = np.linspace(xy[0][i], xy[0][i+1], seg_points)
        y = np.linspace(xy[1][i], xy[1][i+1], seg_points)

        # Create array of linewidths
        linewidths = np.linspace(8, 1, seg_points)

        # Create points array for LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection with varying linewidths and colors
        color_range = colors[counter:counter+seg_points+1]
        lc = LineCollection(segments, linewidths=linewidths, colors=cmap(color_range))
        ax.add_collection(lc)
        counter += seg_points
    ax.scatter(x, y, marker='P', color='orange',s=10)