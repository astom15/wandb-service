from datetime import UTC, datetime, timedelta

import pandas as pd

import wandb

# Initialize W&B run at module level
wandb.init(
    project="bobby-flAI",
    name="validation-metrics",
    job_type="validation",
    config={
        "environment": "production",
        "service_version": "1.0.0"
    }
)

def update_visualizations(validation_data):
    """Update W&B visualizations with new validation data"""
    
    # Convert validation data to pandas DataFrame
    df = pd.DataFrame(validation_data)
    print("DataFrame shape:", df.shape)
    
    # Calculate success rate
    success_rate = df['success'].mean()
    print("Success rate:", success_rate)
    
    # Calculate validation step success rates
    step_success_rates = {
        step: df['validation_steps'].apply(lambda x: x[step]).mean()
        for step in ['array_format', 'markdown_removed', 'intro_text_removed', 'json_parsed', 'structure_validated']
    }
    print("Step success rates:", step_success_rates)
    
    # Prepare error data
    error_data = df[df['success'] == False]
    error_types = {}
    if not error_data.empty:
        error_types = error_data['error_type'].value_counts().to_dict()
    
    # Prepare metrics to log
    metrics = {
        "validation_success_rate": success_rate,
        "validation_steps": step_success_rates,
        "validation_duration": {
            "mean": df['validation_duration_ms'].mean(),
            "p95": df['validation_duration_ms'].quantile(0.95)
        },
        "error_types": error_types
    }
    print("Logging metrics:", metrics)
    
    # Log updated metrics
    wandb.log(metrics)

def create_validation_dashboard():
    """Create a W&B dashboard for validation metrics"""
    # Create initial data for each panel
    wandb.log({
        "validation_success_rate": wandb.plot.line_series(
            xs=[[datetime.now(UTC)]],
            ys=[[0.0]],
            keys=["Success Rate"],
            title="JSON Validation Success Rate Over Time",
            xname="Time"
        ),
        "validation_steps": wandb.plot.bar(
            table=wandb.Table(
                data=[[step, 0.0] for step in ["Array Format", "Markdown Removed", "Intro Text Removed", "JSON Parsed", "Structure Validated"]],
                columns=["Step", "Success Rate"]
            ),
            label="Step",
            value="Success Rate",
            title="Validation Step Success Rates"
        ),
        "validation_duration": wandb.plot.scatter(
            table=wandb.Table(
                data=[[0, 0]],
                columns=["Content Length", "Duration (ms)"]
            ),
            x="Content Length",
            y="Duration (ms)",
            title="Validation Duration vs Content Length"
        ),
        "error_types": wandb.plot.bar(
            table=wandb.Table(
                data=[["No Data", 0]],
                columns=["Error Type", "Count"]
            ),
            label="Error Type",
            value="Count",
            title="Validation Error Types"
        )
    })

if __name__ == "__main__":
    # Create the dashboard
    create_validation_dashboard() 