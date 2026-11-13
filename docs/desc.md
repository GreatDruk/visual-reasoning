One of the hardest problem in modern ML world is to create a system that can handle different modalities by understanding their relationships as well as properties. Such systems are becoming prevalent in different areas, especially when we are trying to solve such problems as:

-   How to find a specific entity using just its text description and vice versa?

In real-world problem we can expand such kind of tasks

-   How to recommend\\find that has only visual representation of something based only on text data?
-   etc.

In order to find a solution we need a system that could handle both modalities understanding their properties simultaneously. You are proposed to solve a task where images have some logical statements that explain relationship between objects on the image. Your goal is to construct such a model that would assess those statements as correct as possible. Each image shows a 3D scene with simple objects. Each caption is a statement about the scene. You need **to predict the probability** that the caption is true for the image. Captions range from simple ("a red cube") to compositional ("there are exactly two objects behind the green sphere").

The most straightforward method is to separately train two encoders for two modalities separately that will be shown for during the lab. Briefly it might look as the following:

```
image   ──► image encoder ──► i (256 numbers)
caption ──► text encoder  ──► t (256 numbers)
score = scale · cos(i, t) + bias  ──► sigmoid ──► probability that the caption is true
```

Still you can find different ways to solve that problem. In this competition you need to design such system that could assess logical statements for 3d shapes on images.