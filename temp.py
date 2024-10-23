from finitewave.tools import Animation2DBuilder


builder = Animation2DBuilder()
builder.write('/Users/arstanbek/Projects/fibrosis-workspace/Finitewave/anim_data',
              shape=(100, 100), shape_scale=10, fps=12)