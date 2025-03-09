import matplotlib.pyplot as plt

def save_plot(data, plot_type, output_path="plot.png"):
    """Sauvegarde un graphique générique."""
    plt.figure(figsize=(8, 6))
    if plot_type == "scatter":
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
    plt.savefig(output_path)
    plt.close()
