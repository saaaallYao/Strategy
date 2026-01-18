import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict
import os


class Visualizer:
    def __init__(self, data_path: str, only_display: bool = False):
        """
        Initialize the Visualizer with trading data

        Args:
            data_path (str): Path to the JSON data file
            only_display (bool): If True, only display graphs without saving
        """
        self.data_path = data_path
        self.only_display = only_display
        self.data = self._load_data()
        self.trades_df = self._process_trades()
        self.generated_plots = {}  # Store plot objects for later saving

    def _load_data(self) -> Dict:
        """Load trading data from JSON file"""
        try:
            with open(self.data_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.data_path}")

    def _process_trades(self) -> pd.DataFrame:
        """Process trades data into a pandas DataFrame"""
        if not self.data.get("trades"):
            return pd.DataFrame()

        trades = []
        for trade in self.data["trades"]:
            trade_data = {
                "timestamp": pd.to_datetime(trade["timestamp"]),
                "action": trade["action"],
                "symbol": trade["symbol"],
                "quantity": trade["quantity"],
                "price": trade["price"],
                "position": trade["position"],
                "value": trade["quantity"] * trade["price"],
            }
            trades.append(trade_data)

        df = pd.DataFrame(trades)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate key performance metrics"""
        if self.trades_df.empty:
            return {
                "sharpe_ratio": 0,
                "annualized_return": 0,
                "max_drawdown": 0,
                "total_pnl": self.data["performance"]["total_pnl"],
            }

        # Calculate returns for each trade
        returns = []
        cumulative_pnl = 0
        pnl_series = [0]  # Start with 0 PnL

        for i, trade in self.trades_df.iterrows():
            if trade["action"] == "SELL":
                # For sells, we gain money
                pnl = trade["value"]
            else:  # BUY
                # For buys, we spend money
                pnl = -trade["value"]

            cumulative_pnl += pnl
            pnl_series.append(cumulative_pnl)

            if i > 0:
                prev_value = pnl_series[i]
                if prev_value != 0:
                    returns.append(pnl / abs(prev_value))

        # Calculate metrics
        if len(returns) > 1:
            returns = np.array(returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0

            # Annualized return (assuming daily returns)
            annualized_return = mean_return * 252  # 252 trading days
        else:
            sharpe_ratio = 0
            annualized_return = 0

        # Max drawdown
        pnl_series = np.array(pnl_series)
        peak = np.maximum.accumulate(pnl_series)
        drawdown = (pnl_series - peak) / np.maximum(peak, 1)
        max_drawdown = np.min(drawdown)

        return {
            "sharpe_ratio": sharpe_ratio,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "total_pnl": self.data["performance"]["total_pnl"],
        }

    def generate_performance_summary(self, save_path: str = None) -> str:
        """Generate performance summary visualization"""
        metrics = self._calculate_performance_metrics()

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Performance Summary - {self.data['strategy']}",
            fontsize=16,
            fontweight="bold",
        )

        # Sharpe Ratio
        ax[0, 0].bar(
            ["Sharpe Ratio"], [metrics["sharpe_ratio"]], color="steelblue", alpha=0.7
        )
        ax[0, 0].set_title("Sharpe Ratio")
        ax[0, 0].set_ylabel("Ratio")
        ax[0, 0].grid(True, alpha=0.3)

        # Annualized Return
        ax[0, 1].bar(
            ["Annualized Return"],
            [metrics["annualized_return"]],
            color="green" if metrics["annualized_return"] > 0 else "red",
            alpha=0.7,
        )
        ax[0, 1].set_title("Annualized Return")
        ax[0, 1].set_ylabel("Return %")
        ax[0, 1].grid(True, alpha=0.3)

        # Max Drawdown
        ax[1, 0].bar(
            ["Max Drawdown"],
            [abs(metrics["max_drawdown"]) * 100],
            color="red",
            alpha=0.7,
        )
        ax[1, 0].set_title("Max Drawdown")
        ax[1, 0].set_ylabel("Drawdown %")
        ax[1, 0].grid(True, alpha=0.3)

        # Total P&L
        ax[1, 1].bar(
            ["Total P&L"],
            [metrics["total_pnl"]],
            color="green" if metrics["total_pnl"] > 0 else "red",
            alpha=0.7,
        )
        ax[1, 1].set_title("Total P&L")
        ax[1, 1].set_ylabel("P&L ($)")
        ax[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.only_display:
            # Store the figure for potential saving later
            self.generated_plots["performance_summary"] = fig
            plt.show()
            return None
        else:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Performance summary saved to: {save_path}")
            else:
                plt.show()

            plt.close()
            return save_path

    def generate_pnl_chart(self, save_path: str = None) -> str:
        """Generate P&L chart with buy/sell points"""
        if self.trades_df.empty:
            print("No trades data available for P&L chart")
            return None

        # Calculate cumulative P&L
        cumulative_pnl = []
        running_pnl = 0

        for _, trade in self.trades_df.iterrows():
            if trade["action"] == "SELL":
                running_pnl += trade["value"]
            else:  # BUY
                running_pnl -= trade["value"]
            cumulative_pnl.append(running_pnl)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot P&L line
        ax.plot(
            self.trades_df["timestamp"],
            cumulative_pnl,
            linewidth=2,
            color="blue",
            label="Cumulative P&L",
        )

        # Add buy/sell points
        for i, (_, trade) in enumerate(self.trades_df.iterrows()):
            color = "green" if trade["action"] == "BUY" else "red"
            marker = "^" if trade["action"] == "BUY" else "v"
            ax.scatter(
                trade["timestamp"],
                cumulative_pnl[i],
                color=color,
                marker=marker,
                s=100,
                alpha=0.7,
                label=f"{trade['action']} {trade['symbol']}",
            )

        # Format the plot
        ax.set_title(
            f"P&L Chart with Buy/Sell Points - {self.data['strategy']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()

        if self.only_display:
            # Store the figure for potential saving later
            self.generated_plots["pnl_chart"] = fig
            plt.show()
            return None
        else:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"P&L chart saved to: {save_path}")
            else:
                plt.show()

            plt.close()
            return save_path

    def generate_trade_analysis(self, save_path: str = None) -> str:
        """Generate individual trade analysis table"""
        if self.trades_df.empty:
            print("No trades data available for trade analysis")
            return None

        # Prepare trade analysis data
        trade_analysis = []
        for i, trade in self.trades_df.iterrows():
            analysis = {
                "Trade #": i + 1,
                "Timestamp": trade["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "Action": trade["action"],
                "Symbol": trade["symbol"],
                "Quantity": f"{trade['quantity']:.6f}",
                "Price": f"${trade['price']:.2f}",
                "Value": f"${trade['value']:.2f}",
                "Position": f"{trade['position']:.6f}",
            }
            trade_analysis.append(analysis)

        # Create figure and table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("tight")
        ax.axis("off")

        # Create table
        table_data = []
        headers = list(trade_analysis[0].keys())
        for trade in trade_analysis:
            table_data.append(list(trade.values()))

        table = ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(1, len(trade_analysis) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        ax.set_title(
            f"Individual Trade Analysis - {self.data['strategy']}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        if self.only_display:
            # Store the figure for potential saving later
            self.generated_plots["trade_analysis"] = fig
            plt.show()
            return None
        else:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Trade analysis saved to: {save_path}")
            else:
                plt.show()

            plt.close()
            return save_path

    def generate_symbol_breakdown(self, save_path: str = None) -> str:
        """Generate symbol-wise breakdown if multiple symbols exist"""
        if self.trades_df.empty:
            print("No trades data available for symbol breakdown")
            return None

        symbols = self.trades_df["symbol"].unique()

        if len(symbols) <= 1:
            print("Only one symbol found, skipping symbol breakdown")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Symbol Breakdown - {self.data['strategy']}",
            fontsize=16,
            fontweight="bold",
        )

        # Trade count by symbol
        symbol_counts = self.trades_df["symbol"].value_counts()
        axes[0, 0].pie(
            symbol_counts.values, labels=symbol_counts.index, autopct="%1.1f%%"
        )
        axes[0, 0].set_title("Trade Count by Symbol")

        # Value traded by symbol
        symbol_values = self.trades_df.groupby("symbol")["value"].sum()
        axes[0, 1].bar(symbol_values.index, symbol_values.values, alpha=0.7)
        axes[0, 1].set_title("Total Value Traded by Symbol")
        axes[0, 1].set_ylabel("Value ($)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Current positions
        positions = self.data["performance"]["positions"]
        axes[1, 0].bar(positions.keys(), positions.values(), alpha=0.7)
        axes[1, 0].set_title("Current Positions")
        axes[1, 0].set_ylabel("Position Size")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Action distribution
        action_counts = self.trades_df["action"].value_counts()
        axes[1, 1].bar(action_counts.index, action_counts.values, alpha=0.7)
        axes[1, 1].set_title("Buy vs Sell Actions")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()

        if self.only_display:
            # Store the figure for potential saving later
            self.generated_plots["symbol_breakdown"] = fig
            plt.show()
            return None
        else:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Symbol breakdown saved to: {save_path}")
            else:
                plt.show()

            plt.close()
            return save_path

    def save_displayed_plots(self, output_dir: str = None) -> Dict[str, str]:
        """Save plots that were previously displayed"""
        if not self.generated_plots:
            print("No plots to save. Generate plots with only_display=True first.")
            return {}

        if output_dir is None:
            output_dir = os.path.dirname(self.data_path)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        for plot_name, fig in self.generated_plots.items():
            file_path = os.path.join(output_dir, f"{plot_name}.png")
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            saved_files[plot_name] = file_path
            print(f"{plot_name} saved to: {file_path}")

            # Close the figure after saving
            plt.close(fig)

        # Clear the stored plots
        self.generated_plots.clear()

        return saved_files

    def clear_displayed_plots(self):
        """Clear displayed plots without saving"""
        for fig in self.generated_plots.values():
            plt.close(fig)
        self.generated_plots.clear()
        print("All displayed plots cleared.")

    def generate_all_graphs(self, output_dir: str = None) -> Dict[str, str]:
        """Generate all available graphs"""
        if output_dir is None:
            output_dir = os.path.dirname(self.data_path) + "/charts"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Generate performance summary
        if not self.only_display:
            perf_path = os.path.join(output_dir, f"performance_summary.png")
            saved_files["performance_summary"] = self.generate_performance_summary(
                perf_path
            )

            # Generate P&L chart
            pnl_path = os.path.join(output_dir, f"pnl_chart.png")
            saved_files["pnl_chart"] = self.generate_pnl_chart(pnl_path)

            # Generate trade analysis
            trade_path = os.path.join(output_dir, f"trade_analysis.png")
            saved_files["trade_analysis"] = self.generate_trade_analysis(trade_path)

            # Generate symbol breakdown (if applicable)
            if len(self.trades_df["symbol"].unique()) > 1:
                symbol_path = os.path.join(output_dir, f"symbol_breakdown.png")
                saved_files["symbol_breakdown"] = self.generate_symbol_breakdown(
                    symbol_path
                )
        else:
            # Only display graphs
            print("Displaying all graphs (only_display=True)...")
            self.generate_performance_summary()
            self.generate_pnl_chart()
            self.generate_trade_analysis()
            if len(self.trades_df["symbol"].unique()) > 1:
                self.generate_symbol_breakdown()

            print(f"\nGenerated {len(self.generated_plots)} plots for review.")
            print(
                "Use save_displayed_plots() to save them or clear_displayed_plots() to discard."
            )

        return saved_files


def main():
    """
    Main function to run the visualizer
    Usage: python visualizer.py <data_file_path> <graph_type>

    Graph types:
    - 'performance': Performance summary with Sharpe ratio, returns, drawdown
    - 'pnl': P&L chart with buy/sell points
    - 'trades': Individual trade analysis table
    - 'symbols': Symbol breakdown (if multiple symbols)
    - 'all': Generate all graphs
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <data_file_path> [graph_type]")
        print("Graph types: performance, pnl, trades, symbols, all")
        return

    data_file_path = sys.argv[1]
    graph_type = sys.argv[2] if len(sys.argv) > 2 else "all"

    try:
        # Initialize visualizer
        visualizer = Visualizer(data_file_path)

        # Generate requested graphs
        if graph_type == "performance":
            visualizer.generate_performance_summary()
        elif graph_type == "pnl":
            visualizer.generate_pnl_chart()
        elif graph_type == "trades":
            visualizer.generate_trade_analysis()
        elif graph_type == "symbols":
            visualizer.generate_symbol_breakdown()
        elif graph_type == "all":
            saved_files = visualizer.generate_all_graphs()
            print("\nAll graphs generated successfully!")
            print("Saved files:")
            for graph_name, file_path in saved_files.items():
                if file_path:
                    print(f"  {graph_name}: {file_path}")
        else:
            print(f"Unknown graph type: {graph_type}")
            print("Available types: performance, pnl, trades, symbols, all")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
