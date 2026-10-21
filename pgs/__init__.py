"""GNN-based N-1 contingency screening (static security assessment)."""
import warnings

# Library deprecation notices (torch.jit, typing) that say nothing about our results.
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_geometric")
