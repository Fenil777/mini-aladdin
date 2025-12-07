"""Mini Aladdin - Portfolio Optimization & Risk Engine CLI."""

import argparse
from pathlib import Path
from src.pipeline import run_full_analysis
from src.reporting import print_analysis_summary, save_summary_csv, plot_efficient_frontier, plot_weights_bar, plot_correlation_heatmap


def main():
    parser = argparse.ArgumentParser(description="Mini Aladdin Portfolio Optimizer")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to data/plots/")
    parser.add_argument("--save-csv", action="store_true", help="Save CSV summary")
    parser.add_argument("--show-plots", action="store_true", help="Show plots in browser")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mini Aladdin - Portfolio Optimization & Risk Engine")
    print("=" * 60)
    print("\nRunning analysis...")
    
    analysis = run_full_analysis(args.config)
    print_analysis_summary(analysis)
    
    if args.save_csv:
        save_summary_csv(analysis)
    
    if args.save_plots or args.show_plots:
        plots_dir = Path("data/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        fig_frontier = plot_efficient_frontier(analysis)
        if args.save_plots:
            fig_frontier.write_html(plots_dir / "efficient_frontier.html")
        if args.show_plots:
            fig_frontier.show()
        
        for portfolio in [analysis.min_variance_portfolio, analysis.max_sharpe_portfolio]:
            fig_weights = plot_weights_bar(portfolio, analysis.asset_names)
            name = portfolio.name.lower().replace(" ", "_")
            if args.save_plots:
                fig_weights.write_html(plots_dir / f"weights_{name}.html")
            if args.show_plots:
                fig_weights.show()
        
        fig_corr = plot_correlation_heatmap(analysis)
        if args.save_plots:
            fig_corr.write_html(plots_dir / "correlation.html")
        if args.show_plots:
            fig_corr.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
