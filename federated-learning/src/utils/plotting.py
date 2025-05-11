def plot_results(all_results):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print("Starting to generate visualizations...")

    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    methods = []
    all_factors = set()

    for method, results in all_results.items():
        methods.append(method)
        for factor in results.keys():
            all_factors.add(factor)

    non_iid_factors = sorted(list(all_factors))

    print(f"Plotting results for {len(methods)} methods across {len(non_iid_factors)} non-IID factors")

    try:
        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D', 'x']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, method in enumerate(methods):
            method_factors = []
            method_accuracies = []

            for factor in non_iid_factors:
                if factor in all_results[method]:
                    method_factors.append(factor)
                    method_accuracies.append(all_results[method][factor]['test_accuracy'])

            if method_factors and method_accuracies:
                plt.plot(method_factors, method_accuracies, marker=markers[i % len(markers)], label=method, color=colors[i % len(colors)])

        plt.xlabel('Non-IID Factor', fontsize=14)
        plt.ylabel('Test Accuracy', fontsize=14)
        plt.title('Test Accuracy Comparison Across Aggregation Methods', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['results_dir'], 'test_accuracy_comparison.png'), dpi=300)
        plt.savefig(os.path.join(CONFIG['results_dir'], 'test_accuracy_comparison.pdf'))
        plt.close()
        print("Test accuracy comparison plot created successfully")
    except Exception as e:
        print(f"Error creating test accuracy plot: {e}")

    try:
        plt.figure(figsize=(14, 10))
        width = 0.15
        x = np.arange(len(non_iid_factors))

        methods_with_data = 0

        for i, method in enumerate(methods):
            method_accuracies = []
            valid_factors = []

            for j, factor in enumerate(non_iid_factors):
                if factor in all_results[method]:
                    method_accuracies.append(all_results[method][factor]['test_accuracy'])
                    valid_factors.append(factor)

            if method_accuracies and valid_factors:
                bar_positions = [x[j] + i * width for j in range(len(valid_factors))]
                plt.bar(bar_positions, method_accuracies, width, label=method, color=colors[i % len(colors)])
                methods_with_data += 1

        if methods_with_data > 0:
            plt.xlabel('Non-IID Factor', fontsize=14)
            plt.ylabel('Test Accuracy', fontsize=14)
            plt.title('Test Accuracy Comparison (Bar Chart)', fontsize=16, fontweight='bold')
            plt.xticks(x + width * (methods_with_data - 1) / 2, non_iid_factors)
            plt.legend(fontsize=12)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['results_dir'], 'test_accuracy_bar_chart.png'), dpi=300)
            plt.savefig(os.path.join(CONFIG['results_dir'], 'test_accuracy_bar_chart.pdf'))
            plt.close()
            print("Test accuracy bar chart created successfully")
    except Exception as e:
        print(f"Error creating bar chart: {e}")

    for method in methods:
        for factor in non_iid_factors:
            if factor not in all_results.get(method, {}):
                continue

            metrics_data = all_results[method].get(factor, {}).get('metrics', {})
            if not metrics_data:
                continue

            try:
                required_metrics = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss']
                if not all(key in metrics_data for key in required_metrics):
                    continue

                lengths = [len(metrics_data[key]) for key in required_metrics]
                if len(set(lengths)) > 1:
                    continue

                plt.figure(figsize=(12, 8))

                plt.subplot(2, 1, 1)
                train_accuracy = metrics_data['train_accuracy']
                val_accuracy = metrics_data['val_accuracy']
                epochs = range(len(train_accuracy))

                plt.plot(epochs, train_accuracy, label='Train Accuracy', color='#1f77b4', linewidth=2)
                plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
                plt.xlabel('Round', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.title(f'{method} - Non-IID Factor {factor} - Accuracy', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True)

                plt.subplot(2, 1, 2)
                train_loss = metrics_data['train_loss']
                val_loss = metrics_data['val_loss']

                plt.plot(epochs, train_loss, label='Train Loss', color='#2ca02c', linewidth=2)
                plt.plot(epochs, val_loss, label='Validation Loss', color='#d62728', linewidth=2)
                plt.xlabel('Round', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title(f'{method} - Non-IID Factor {factor} - Loss', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(CONFIG['results_dir'], f'learning_curves_{method.lower()}_noniid{factor}.png'), dpi=300)
                plt.savefig(os.path.join(CONFIG['results_dir'], f'learning_curves_{method.lower()}_noniid{factor}.pdf'))
                plt.close()
                print(f"Learning curves for {method} with factor {factor} created successfully")
            except Exception as e:
                print(f"Error creating learning curves for {method} with factor {factor}: {e}")

    try:
        summary_data = []
        for method in methods:
            for factor in non_iid_factors:
                if factor not in all_results.get(method, {}):
                    continue

                method_data = all_results[method].get(factor, {})
                metrics_data = method_data.get('metrics', {})

                if 'test_accuracy' in method_data and 'round_times' in metrics_data:
                    summary_data.append({
                        'Method': method,
                        'Non-IID Factor': factor,
                        'Test Accuracy': f"{method_data['test_accuracy']:.4f}",
                        'Avg Round Time (s)': f"{np.mean(metrics_data['round_times']):.2f}"
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
            ax.axis('off')

            table = ax.table(
                cellText=summary_df.values,
                colLabels=summary_df.columns,
                loc='center',
                cellLoc='center'
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            plt.title('Federated Learning Results Summary', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['results_dir'], 'results_table.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(CONFIG['results_dir'], 'results_table.pdf'), bbox_inches='tight')
            plt.close()
            print("Results summary table created successfully")
    except Exception as e:
        print(f"Error creating summary table: {e}")

    print("Visualization generation completed")