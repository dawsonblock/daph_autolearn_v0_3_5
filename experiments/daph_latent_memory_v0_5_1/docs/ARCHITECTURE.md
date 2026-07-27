# Architecture

```
MathWorkingState
      |
      v
StateTokenEncoder
      |
      v
K latent tokens ---------+
                         |
Question -> embeddings -> concat -> frozen transformer -> answer
                         |
                         +-> hidden-state alignment target
```

The base transformer is immutable during the experiment.

The only trainable component is the latent state encoder.

This deliberately isolates the scientific question: whether structured external process state can be
compiled into a latent representation that the frozen model can exploit.

## Future integration

After qualification:

- GDN2: fast recurrent state
- latent memory: information-rich task state
- external memory: persistent canonical knowledge
- executive controller: retrieve / reason / verify / stop policy
- AutoLearn: outcome-driven policy and memory adaptation
