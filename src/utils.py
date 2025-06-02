import numpy as np
import matplotlib.pyplot as plt


def plot_contours_with_path(f, xlim, ylim, title, paths=None, method_names=None, levels=20):
    """
    Plot contour lines of a 2D function with optional optimization paths.

    Parameters:
    - f: objective function that returns (value, gradient, hessian)
    - xlim: tuple (xmin, xmax) for x-axis limits
    - ylim: tuple (ymin, ymax) for y-axis limits
    - title: plot title
    - paths: list of paths, where each path is a list of [x, y] points
    - method_names: list of method names corresponding to paths
    - levels: number of contour levels or specific levels array
    """

    # Create a grid for contour plot
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Evaluate function values on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j], _, _ = f(point, False)  # Don't need Hessian for plotting

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot contour lines
    if isinstance(levels, int):
        # Automatically choose levels based on function values
        contour = plt.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
    else:
        # Use specific levels
        contour = plt.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)

    plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    # Plot optimization paths if provided
    if paths is not None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']

        for i, path in enumerate(paths):
            if len(path) > 0:
                # Convert path to numpy array for easier indexing
                path_array = np.array(path)
                x_path = path_array[:, 0]
                y_path = path_array[:, 1]

                # Choose color and marker
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                # Plot path line
                plt.plot(x_path, y_path, color=color, linewidth=2, alpha=0.8)

                # Plot points along the path
                plt.plot(x_path, y_path, marker=marker, color=color,
                         markersize=6, markerfacecolor='white', markeredgewidth=2,
                         linestyle='', alpha=0.9)

                # Mark starting point
                plt.plot(x_path[0], y_path[0], marker='*', color=color,
                         markersize=12, label=f'{method_names[i] if method_names else f"Method {i + 1}"} start')

                # Mark ending point
                plt.plot(x_path[-1], y_path[-1], marker='X', color=color,
                         markersize=10, markeredgewidth=3)

    # Set labels and title
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Add legend if paths are plotted
    if paths is not None and method_names is not None:
        plt.legend(fontsize=10)

    plt.tight_layout()
    return plt.gcf()


def plot_function_values(iterations_list, values_list, method_names, title):
    """
    Plot function values vs iteration number for comparison of methods.

    Parameters:
    - iterations_list: list of iteration arrays, one for each method
    - values_list: list of function value arrays, one for each method
    - method_names: list of method names
    - title: plot title
    """

    plt.figure(figsize=(10, 6))

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']

    for i, (iterations, values, name) in enumerate(zip(iterations_list, values_list, method_names)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        plt.semilogy(iterations, values, color=color, linestyle=linestyle,
                     linewidth=2, marker='o', markersize=4, label=name)

    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Function Value (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    return plt.gcf()


def save_plot(fig, filename):
    """
    Save a matplotlib figure to file.

    Parameters:
    - fig: matplotlib figure object
    - filename: output filename
    """
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")


def print_final_result(method_name, final_x, final_f, success, num_iterations):
    """
    Print formatted final results of optimization.

    Parameters:
    - method_name: name of the optimization method
    - final_x: final x location
    - final_f: final function value
    - success: success flag
    - num_iterations: number of iterations performed
    """
    print(f"\n{'=' * 50}")
    print(f"FINAL RESULTS - {method_name.upper()}")
    print(f"{'=' * 50}")
    print(f"Final location: x = {final_x}")
    print(f"Final objective value: f(x) = {final_f:.12f}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Success: {'YES' if success else 'NO'}")
    print(f"{'=' * 50}\n")