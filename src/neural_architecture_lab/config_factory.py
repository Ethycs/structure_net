
import argparse
from datetime import datetime
from .lab import LabConfig

class LabConfigFactory:
    @staticmethod
    def create_from_args(description="Neural Architecture Lab Experiment"):
        parser = argparse.ArgumentParser(description=description)
        
        # General NAL config
        parser.add_argument('--project-name', type=str, default="nal_experiment", help="Name of the project.")
        parser.add_argument('--results-dir', type=str, help="Directory to save results.")
        parser.add_argument('--num-workers', type=int, default=4, help="Number of parallel workers.")
        parser.add_argument('--num-gpus', type=int, default=1, help="Number of GPUs to use.")

        # Logging config
        parser.add_argument('--log-level', type=str, default='INFO', help='Global log level.')
        parser.add_argument('--log-to-file', action='store_true', help='Enable logging to a file.')
        parser.add_argument('--log-to-chroma', action='store_true', help='Enable logging to ChromaDB.')
        parser.add_argument('--log-to-wandb', action='store_true', help='Enable logging to Weights & Biases.')
        parser.add_argument('--disable-wandb', action='store_true', help='Explicitly disable W&B logging.')
        parser.add_argument('--wandb-project', type=str, help='W&B project name.')
        parser.add_argument('--module-log-level', nargs='*', help='Set log level for specific modules, e.g., "data_factory:DEBUG".')

        # Experiment-specific config
        parser.add_argument('--num-hypotheses', type=int, default=10, help='Number of hypotheses to test.')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
        parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')

        args = parser.parse_args()

        results_dir = args.results_dir or f"nal_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return LabConfig(
            project_name=args.wandb_project or args.project_name,
            results_dir=results_dir,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            enable_wandb=args.log_to_wandb and not args.disable_wandb,
            log_level=args.log_level.upper(),
            log_to_file=args.log_to_file,
            log_to_chroma=args.log_to_chroma,
            module_log_levels=dict(item.split(':') for item in args.module_log_level) if args.module_log_level else None,
            # Pass experiment-specific args through a separate dict
            experiment_params={
                'num_hypotheses': args.num_hypotheses,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            }
        )
