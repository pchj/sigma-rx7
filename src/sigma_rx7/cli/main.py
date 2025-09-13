"""Command Line Interface for SigmaRx7."""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import click
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

from ..core.config import Config, load_config
from ..core.pipeline import SigmaRx7Pipeline

# Setup structured logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
console = Console()


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """SigmaRx7 - Synthetic healthcare ETL/ELT pipeline for FHIR data processing."""
    
    # Setup logging level
    if verbose:
        structlog.configure(
            processors=[
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    
    # Display banner
    if ctx.invoked_subcommand != 'version':
        console.print(Panel.fit(
            "[bold blue]SigmaRx7[/bold blue]\n"
            "Synthetic healthcare ETL/ELT pipeline for FHIR data processing\n"
            "Version 0.1.0",
            style="cyan"
        ))


@cli.command()
def version():
    """Show version information."""
    console.print("SigmaRx7 version 0.1.0")


@cli.command()
@click.option('--input', '-i', required=True, help='Input data path')
@click.option('--output', '-o', default='output', help='Output directory')
@click.option('--format', '-f', default='synthea', type=click.Choice(['synthea', 'forgerx', 'fhir_bundle']), 
              help='Input data format')
@click.pass_context
def run(ctx, input: str, output: str, format: str):
    """Run the complete SigmaRx7 pipeline."""
    config = ctx.obj['config']
    
    console.print(f"[bold green]Starting SigmaRx7 pipeline[/bold green]")
    console.print(f"Input: {input}")
    console.print(f"Output: {output}")
    console.print(f"Format: {format}")
    
    async def run_pipeline():
        pipeline = SigmaRx7Pipeline(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running pipeline...", total=None)
            
            try:
                result = await pipeline.run_full_pipeline(input, output, format)
                progress.update(task, description="Pipeline completed!")
                
                # Display results
                _display_pipeline_results(result)
                
            except Exception as e:
                progress.update(task, description=f"Pipeline failed: {e}")
                console.print(f"[bold red]Error:[/bold red] {e}")
                sys.exit(1)
            finally:
                pipeline.close()
    
    asyncio.run(run_pipeline())


@cli.command()
@click.option('--synthea-path', required=True, help='Path to Synthea data')
@click.pass_context
def load_synthea(ctx, synthea_path: str):
    """Load Synthea synthetic data."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Loading Synthea data from:[/bold blue] {synthea_path}")
    
    async def load_data():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = pipeline.load_synthea_data(synthea_path)
            
            console.print("[bold green]Synthea data loaded successfully![/bold green]")
            console.print(f"Patients processed: {result.get('patients_processed', 0)}")
            console.print(f"Medications processed: {result.get('medications_processed', 0)}")
            
        except Exception as e:
            console.print(f"[bold red]Error loading Synthea data:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(load_data())


@cli.command()
@click.option('--forgerx-path', required=True, help='Path to ForgeRx data')
@click.pass_context
def load_forgerx(ctx, forgerx_path: str):
    """Load ForgeRx synthetic data."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Loading ForgeRx data from:[/bold blue] {forgerx_path}")
    
    async def load_data():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = pipeline.load_forgerx_data(forgerx_path)
            
            console.print("[bold green]ForgeRx data loaded successfully![/bold green]")
            console.print(f"Patients processed: {result.get('patients_processed', 0)}")
            console.print(f"Medications processed: {result.get('medications_processed', 0)}")
            
        except Exception as e:
            console.print(f"[bold red]Error loading ForgeRx data:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(load_data())


@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status and statistics."""
    config = ctx.obj['config']
    
    async def get_status():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            status_info = pipeline.get_pipeline_status()
            _display_status(status_info)
        except Exception as e:
            console.print(f"[bold red]Error getting status:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(get_status())


@cli.command()
@click.pass_context
def init_db(ctx):
    """Initialize database schema."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Initializing database schema...[/bold blue]")
    
    async def init_database():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            pipeline.create_database_schema()
            console.print("[bold green]Database schema initialized successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error initializing database:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(init_database())


@cli.command()
@click.option('--output', '-o', default='config.yaml', help='Output configuration file path')
def init_config(output: str):
    """Initialize a new configuration file."""
    config = Config()
    config.to_yaml(output)
    console.print(f"[bold green]Configuration file created:[/bold green] {output}")
    console.print("Edit the configuration file and run the pipeline with: sigma-rx7 -c config.yaml run")


@cli.group()
def analyze():
    """Analysis commands."""
    pass


@analyze.command()
@click.pass_context
def overlaps(ctx):
    """Run medication overlap analysis."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Running medication overlap analysis...[/bold blue]")
    
    async def run_overlap_analysis():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = await pipeline.overlap_detector.detect_overlaps()
            _display_overlap_results(result)
        except Exception as e:
            console.print(f"[bold red]Error in overlap analysis:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(run_overlap_analysis())


@analyze.command()
@click.pass_context
def generics(ctx):
    """Run generic medication recommendations."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Running generic medication analysis...[/bold blue]")
    
    async def run_generic_analysis():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = await pipeline.generic_recommender.recommend_generics()
            _display_generic_results(result)
        except Exception as e:
            console.print(f"[bold red]Error in generic analysis:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(run_generic_analysis())


@analyze.command()
@click.pass_context
def payer_rules(ctx):
    """Run payer rules analysis."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Running payer rules analysis...[/bold blue]")
    
    async def run_payer_analysis():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = await pipeline.payer_rules.apply_rules()
            _display_payer_results(result)
        except Exception as e:
            console.print(f"[bold red]Error in payer rules analysis:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(run_payer_analysis())


@analyze.command()
@click.pass_context
def predict_denials(ctx):
    """Run ML-based denial prediction."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Running denial prediction analysis...[/bold blue]")
    
    async def run_prediction():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            if pipeline.denial_predictor:
                result = await pipeline.denial_predictor.predict_denials()
                _display_prediction_results(result)
            else:
                console.print("[yellow]ML prediction is disabled in configuration[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error in denial prediction:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(run_prediction())


@cli.group()
def export():
    """Export commands."""
    pass


@export.command()
@click.option('--output', '-o', default='output', help='Output directory')
@click.pass_context
def fhir(ctx, output: str):
    """Export FHIR bundles."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Exporting FHIR bundles to:[/bold blue] {output}")
    
    async def export_fhir():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = await pipeline.fhir_exporter.export_bundles(output)
            console.print(f"[bold green]FHIR export completed![/bold green]")
            console.print(f"Bundles exported: {result.get('bundles_exported', 0)}")
            console.print(f"Total resources: {result.get('total_resources', 0)}")
        except Exception as e:
            console.print(f"[bold red]Error exporting FHIR:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(export_fhir())


@export.command()
@click.option('--output', '-o', default='output', help='Output directory')
@click.pass_context
def features(ctx, output: str):
    """Export feature tables."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Exporting feature tables to:[/bold blue] {output}")
    
    async def export_features():
        pipeline = SigmaRx7Pipeline(config)
        
        try:
            result = await pipeline.feature_exporter.export_features(output)
            console.print(f"[bold green]Feature export completed![/bold green]")
            console.print(f"Tables exported: {result.get('tables_exported', 0)}")
            console.print(f"Total rows: {result.get('total_rows', 0)}")
        except Exception as e:
            console.print(f"[bold red]Error exporting features:[/bold red] {e}")
            sys.exit(1)
        finally:
            pipeline.close()
    
    asyncio.run(export_features())


def _display_pipeline_results(result: dict):
    """Display pipeline execution results."""
    console.print("\n[bold green]Pipeline Results:[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Details")
    
    for stage, stage_result in result.get("stages", {}).items():
        status = "✅ Success" if isinstance(stage_result, dict) else "❌ Failed"
        details = ""
        
        if isinstance(stage_result, dict):
            if "patients_processed" in stage_result:
                details += f"Patients: {stage_result['patients_processed']} "
            if "medications_processed" in stage_result:
                details += f"Medications: {stage_result['medications_processed']} "
            if "total_overlaps" in stage_result:
                details += f"Overlaps: {stage_result['total_overlaps']} "
            if "total_recommendations" in stage_result:
                details += f"Recommendations: {stage_result['total_recommendations']} "
        
        table.add_row(stage.replace("_", " ").title(), status, details)
    
    console.print(table)


def _display_status(status_info: dict):
    """Display pipeline status information."""
    console.print("\n[bold blue]Pipeline Status:[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")
    
    db_status = "✅ Connected" if status_info.get("database_connected") else "❌ Disconnected"
    table.add_row("Database", db_status, "")
    
    for table_name, count in status_info.get("tables", {}).items():
        table.add_row(f"Table: {table_name}", "📊 Data", f"{count} records")
    
    console.print(table)


def _display_overlap_results(result: dict):
    """Display overlap analysis results."""
    console.print(f"\n[bold green]Overlap Analysis Results:[/bold green]")
    console.print(f"Total patients: {result.get('total_patients', 0)}")
    console.print(f"Patients with overlaps: {result.get('patients_with_overlaps', 0)}")
    console.print(f"Total overlaps found: {result.get('total_overlaps', 0)}")
    
    if result.get('overlaps_by_severity'):
        console.print("\nOverlaps by severity:")
        for severity, count in result['overlaps_by_severity'].items():
            console.print(f"  {severity}: {count}")


def _display_generic_results(result: dict):
    """Display generic recommendation results."""
    console.print(f"\n[bold green]Generic Recommendation Results:[/bold green]")
    console.print(f"Total patients: {result.get('total_patients', 0)}")
    console.print(f"Patients with recommendations: {result.get('patients_with_recommendations', 0)}")
    console.print(f"Total recommendations: {result.get('total_recommendations', 0)}")
    console.print(f"Potential annual savings: ${result.get('potential_annual_savings', 0):.2f}")


def _display_payer_results(result: dict):
    """Display payer rules analysis results."""
    console.print(f"\n[bold green]Payer Rules Analysis Results:[/bold green]")
    console.print(f"Total patients: {result.get('total_patients', 0)}")
    console.print(f"Total violations: {result.get('total_violations', 0)}")
    console.print(f"Estimated denials: {result.get('estimated_denials', 0)}")
    
    if result.get('violations_by_type'):
        console.print("\nViolations by type:")
        for violation_type, count in result['violations_by_type'].items():
            console.print(f"  {violation_type}: {count}")


def _display_prediction_results(result: dict):
    """Display denial prediction results."""
    console.print(f"\n[bold green]Denial Prediction Results:[/bold green]")
    console.print(f"Total predictions: {result.get('total_predictions', 0)}")
    console.print(f"High risk: {result.get('high_risk_count', 0)}")
    console.print(f"Medium risk: {result.get('medium_risk_count', 0)}")
    console.print(f"Low risk: {result.get('low_risk_count', 0)}")
    console.print(f"Average denial probability: {result.get('average_denial_probability', 0):.3f}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()