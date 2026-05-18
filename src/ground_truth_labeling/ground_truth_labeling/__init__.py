"""
Ground truth labeling toolkit for phantom obstacle detection validation.

Provides tools to:
1. Extract frames from rosbags
2. Generate labeling interface
3. Analyze and validate phantom rate claims

Main workflow:
    from labeling_workflow import GroundTruthWorkflow
    workflow = GroundTruthWorkflow(...)
    workflow.run()
"""

__version__ = "0.1.0"
