import time
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
    print(df)
    
    # Calculate success rate
    success_rate = df['success'].mean()
    print("Success rate:", success_rate)
    
    # Calculate validation step success rates
    step_success_rates = {
        step: df['validation_steps'].apply(lambda x: x.get(step,False)).mean()
        for step in ['array_format', 'markdown_removed', 'intro_text_removed', 'json_parsed', 'structure_validated']
    }
    print("Step success rates:", step_success_rates)
    
    # Prepare error data
    error_data = df[df['success'] == False]
    error_types = {}
    if not error_data.empty:
        error_types = error_data['error_type'].value_counts().to_dict()
    
    # Log metrics to existing tables
    metrics_to_log = {
        "validation_success_rate": success_rate,
        "validation_duration/mean_ms": df['validation_duration_ms'].mean(),
        "validation_duration/p95_ms": df['validation_duration_ms'].quantile(0.95),
    }
    if error_types:
        for error_type, count in error_types.items():
            metrics_to_log[f"error_types/{error_type}"] = count
        metrics_to_log["error_types/total_errors"] = sum(error_types.values())
    else:
        for error_type in ['ValueError', 'JSONDecodeError', 'KeyError', 'IndexError', 'TypeError', 'AttributeError']:
            metrics_to_log[f"error_types/{error_type}"] = 0
    
    step_success_table = wandb.Table(
        data=[[step.replace('_', ' ').title(), rate] for step, rate in step_success_rates.items()],
        columns=["Step", "Success Rate"]
    )   
    scatter_data = df[['content_length', 'validation_duration_ms']].values.tolist()
    duration_scatter_table = wandb.Table(
        data=scatter_data,
        columns=["Content Length", "Duration (ms)"]
    )
    
    # Create a table for validation metadata
    metadata_table = wandb.Table(
        data=[[row['session_id'], row['trace_id'], row['timestamp']] for _, row in df.iterrows()],
        columns=["Session ID", "Trace ID", "Timestamp"]
    )
    
    wandb.log({
        **metrics_to_log,
        "validation_steps_bar_chart": wandb.plot.bar(
            table=step_success_table,
            label="Step",
            value="Success Rate",
            title="Validation Step Success Rates"
        ),
        "validation_duration_scatter_chart": wandb.plot.scatter(
            table=duration_scatter_table,
            x="Content Length",
            y="Duration (ms)",
            title="Validation Duration vs Content Length"
        ),
        "error_types_bar_chart": wandb.plot.bar(
            table=wandb.Table(
                data=[[k,v] for k,v in error_types.items()],
                columns=["Error Type", "Count"]
            ),
            label="Error Type",
            value="Count",
            title="Validation Error Types"
        ),
        "validation_metadata": metadata_table,
        "validation_raw_data_table": wandb.Table(dataframe=df)
    })
    print("Logged metrics to W&B tables")

def test_visualization_logging():
    
    """Test function to verify visualization data is being logged correctly"""
    # Create test validation data with both success and failure cases
    test_data_1 = [
        {
            "success": True,
            "validation_steps": {
                "array_format": True,
                "markdown_removed": True,
                "intro_text_removed": True,
                "json_parsed": True,
                "structure_validated": True
            },
            "validation_duration_ms": 100,
            "content_length": 500,
            "recipe_count": 2,
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": "test-session-1",
            "trace_id": "test-trace-1"
        },
        {
            "success": False,
            "validation_steps": {
                "array_format": True,
                "markdown_removed": True,
                "intro_text_removed": True,
                "json_parsed": False,
                "structure_validated": False
            },
            "validation_duration_ms": 150,
            "content_length": 600,
            "error_type": "ValueError",
            "error_message": "Invalid JSON format",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": "test-session-2",
            "trace_id": "test-trace-2"
        }
    ]
    test_data_2 = [
        {
            "success": True,
            "validation_steps": {
                "array_format": True,
                "markdown_removed": True,
                "intro_text_removed": True,
                "json_parsed": True,
                "structure_validated": True
            },
            "validation_duration_ms": 90,
            "content_length": 480,
            "recipe_count": 3,
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": "test-session-3",
            "trace_id": "test-trace-3"
        },
        {
            "success": False,
            "validation_steps": {
                "array_format": False,
                "markdown_removed": True,
                "intro_text_removed": True,
                "json_parsed": True,
                "structure_validated": True
            },
            "validation_duration_ms": 110,
            "content_length": 550,
            "error_type": "FormatError",
            "error_message": "Array format issue",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": "test-session-4",
            "trace_id": "test-trace-4"
        },
        {
            "success": False,
            "validation_steps": {
                "array_format": True,
                "markdown_removed": False,
                "intro_text_removed": True,
                "json_parsed": True,
                "structure_validated": True
            },
            "validation_duration_ms": 120,
            "content_length": 580,
            "error_type": "MarkdownError",
            "error_message": "Markdown present",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": "test-session-5",
            "trace_id": "test-trace-5"
        }
    ]
    
    print("\nTesting visualization logging with first dataset...")
    # Update visualizations with test data
    update_visualizations(test_data_1)
    time.sleep(2)
    print("\nTesting visualization logging with second dataset...")
    update_visualizations(test_data_2)
    
    print("\nTest complete. Check W&B UI for updated visualizations.")

if __name__ == "__main__":
    
    test_visualization_logging() 
    wandb.finish()