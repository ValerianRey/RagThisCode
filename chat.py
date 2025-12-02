import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


def print_messages(response):
    messages = response["messages"]

    for message in messages:

        if isinstance(message, HumanMessage):
            emoji = "👤"
            content = message.content
        elif isinstance(message, AIMessage):
            emoji = "🤖"
            if message.tool_calls:
                content = message.tool_calls[0]["args"]
            else:
                content = message.content
        elif isinstance(message, ToolMessage):
            emoji = "🔧"
            content = json.loads(message.content)
            content = f"\n{'=' * 100}\n".join(content)
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        print(f"{emoji} {content}")


async def main():

    server_ip = "54.36.102.143"  # "54.36.102.143"

    client = MultiServerMCPClient(
        {
            "similarity_search": {
                "url": f"http://{server_ip}:9000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()

    SYSTEM_PROMPT = r"""
    You are a helpful assistant that is specifically designed to answer questions about the TorchJD codebase, whose content is stored in a vector store.
    Please only answer questions about the codebase by running the similarity_search tool.
    If necessary, you can call the tool multiple times to get the information you need.
    Also use markdown to format your responses.

    You only have access to the .py files in the codebase, so you can't answer questions about the other files.
    Here is the README.md file of TorchJD:

    TorchJD is a library extending autograd to enable
    [Jacobian descent](https://arxiv.org/pdf/2406.16232) with PyTorch. It can be used to train neural
    networks with multiple objectives. In particular, it supports multi-task learning, with a wide
    variety of aggregators from the literature. It also enables the instance-wise risk minimization
    paradigm. The full documentation is available at [torchjd.org](https://torchjd.org), with several
    usage examples.

    ## Jacobian descent (JD)
    Jacobian descent is an extension of gradient descent supporting the optimization of vector-valued
    functions. This algorithm can be used to train neural networks with multiple loss functions. In this
    context, JD iteratively updates the parameters of the model using the Jacobian matrix of the vector
    of losses (the matrix stacking each individual loss' gradient). For more details, please refer to
    Section 2.1 of the [paper](https://arxiv.org/pdf/2406.16232).

    ### How does this compare to averaging the different losses and using gradient descent?

    Averaging the losses and computing the gradient of the mean is mathematically equivalent to
    computing the Jacobian and averaging its rows. However, this approach has limitations. If two
    gradients are conflicting (they have a negative inner product), simply averaging them can result in
    an update vector that is conflicting with one of the two gradients. Averaging the losses and making
    a step of gradient descent can thus lead to an increase of one of the losses.

    This is illustrated in the following picture, in which the two objectives' gradients $g_1$ and $g_2$
    are conflicting, and averaging them gives an update direction that is detrimental to the first
    objective. Note that in this picture, the dual cone, represented in green, is the set of vectors
    that have a non-negative inner product with both $g_1$ and $g_2$.

    ![image](docs/source/_static/direction_upgrad_mean.svg)

    With Jacobian descent, $g_1$ and $g_2$ are computed individually and carefully aggregated using an
    aggregator $\mathcal A$. In this example, the aggregator is the Unconflicting Projection of
    Gradients $\mathcal A_{\text{UPGrad}}$: it
    projects each gradient onto the dual cone, and averages the projections. This ensures that the
    update will always be beneficial to each individual objective (given a sufficiently small step
    size). In addition to $\mathcal A_{\text{UPGrad}}$, TorchJD supports
    [more than 10 aggregators from the literature](https://torchjd.org/stable/docs/aggregation).

    ## Installation
    <!-- start installation -->
    TorchJD can be installed directly with pip:
    ```bash
    pip install torchjd
    ```
    <!-- end installation -->
    Some aggregators may have additional dependencies. Please refer to the
    [installation documentation](https://torchjd.org/stable/installation) for them.

    ## Usage
    There are two main ways to use TorchJD. The first one is to replace the usual call to
    `loss.backward()` by a call to
    [`torchjd.autojac.backward`](https://torchjd.org/stable/docs/autojac/backward/) or
    [`torchjd.autojac.mtl_backward`](https://torchjd.org/stable/docs/autojac/mtl_backward/), depending
    on the use-case. This will compute the Jacobian of the vector of losses with respect to the model
    parameters, and aggregate it with the specified
    [`Aggregator`](https://torchjd.org/stable/docs/aggregation/index.html#torchjd.aggregation.Aggregator).
    Whenever you want to optimize the vector of per-sample losses, you should rather use the
    [`torchjd.autogram.Engine`](https://torchjd.org/stable/docs/autogram/engine.html). Instead of
    computing the full Jacobian at once, it computes the Gramian of this Jacobian, layer by layer, in a
    memory-efficient way. A vector of weights (one per element of the batch) can then be extracted from
    this Gramian, using a
    [`Weighting`](https://torchjd.org/stable/docs/aggregation/index.html#torchjd.aggregation.Weighting),
    and used to combine the losses of the batch. Assuming each element of the batch is
    processed independently from the others, this approach is equivalent to
    [`torchjd.autojac.backward`](https://torchjd.org/stable/docs/autojac/backward/) while being
    generally much faster due to the lower memory usage. Note that we're still working on making
    `autogram` faster and more memory-efficient, and it's interface may change in future releases.

    The following example shows how to use TorchJD to train a multi-task model with Jacobian descent,
    using [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/).

    ```diff
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    + from torchjd.autojac import mtl_backward
    + from torchjd.aggregation import UPGrad

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    loss_fn = MSELoss()
    optimizer = SGD(params, lr=0.1)
    + aggregator = UPGrad()

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)
        output1 = task1_module(features)
        output2 = task2_module(features)
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)

        optimizer.zero_grad()
    -     loss = loss1 + loss2
    -     loss.backward()
    +     mtl_backward(losses=[loss1, loss2], features=features, aggregator=aggregator)
        optimizer.step()
    ```

    > [!NOTE]
    > In this example, the Jacobian is only with respect to the shared parameters. The task-specific
    > parameters are simply updated via the gradient of their task’s loss with respect to them.

    The following example shows how to use TorchJD to minimize the vector of per-instance losses with
    Jacobian descent using [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/).

    ```diff
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    + from torchjd.autogram import Engine
    + from torchjd.aggregation import UPGradWeighting

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU(), Linear(3, 1), ReLU())

    - loss_fn = MSELoss()
    + loss_fn = MSELoss(reduction="none")
    optimizer = SGD(model.parameters(), lr=0.1)

    + weighting = UPGradWeighting()
    + engine = Engine(model, batch_dim=0)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task

    for input, target in zip(inputs, targets):
        output = model(input).squeeze(dim=1)  # shape [16]
    -     loss = loss_fn(output, target)  # shape [1]
    +     losses = loss_fn(output, target)  # shape [16]

        optimizer.zero_grad()
    -     loss.backward()
    +     gramian = engine.compute_gramian(losses)  # shape: [16, 16]
    +     weights = weighting(gramian)  # shape: [16]
    +     losses.backward(weights)
        optimizer.step()
    ```

    Lastly, you can even combine the two approaches by considering multiple tasks and each element of
    the batch independently. We call that Instance-Wise Multitask Learning (IWMTL).

    ```python
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import Flattening, UPGradWeighting
    from torchjd.autogram import Engine

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    optimizer = SGD(params, lr=0.1)
    mse = MSELoss(reduction="none")
    weighting = Flattening(UPGradWeighting())
    engine = Engine(shared_module, batch_dim=0)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)  # shape: [16, 3]
        out1 = task1_module(features).squeeze(1)  # shape: [16]
        out2 = task2_module(features).squeeze(1)  # shape: [16]

        # Compute the matrix of losses: one loss per element of the batch and per task
        losses = torch.stack([mse(out1, target1), mse(out2, target2)], dim=1)  # shape: [16, 2]

        # Compute the gramian (inner products between pairs of gradients of the losses)
        gramian = engine.compute_gramian(losses)  # shape: [16, 2, 2, 16]

        # Obtain the weights that lead to no conflict between reweighted gradients
        weights = weighting(gramian)  # shape: [16, 2]

        optimizer.zero_grad()
        # Do the standard backward pass, but weighted using the obtained weights
        losses.backward(weights)
        optimizer.step()
    ```

    > [!NOTE]
    > Here,  because the losses are a matrix instead of a simple vector, we compute a *generalized
    > Gramian* and we extract weights from it using a
    > [GeneralizedWeighting](https://torchjd.org/docs/aggregation/index.html#torchjd.aggregation.GeneralizedWeighting).

    More usage examples can be found [here](https://torchjd.org/stable/examples/).

    ## Supported Aggregators and Weightings
    TorchJD provides many existing aggregators from the literature, listed in the following table.

    <!-- recommended aggregators first, then alphabetical order -->
    | Aggregator                                                                                                 | Weighting                                                                                                              | Publication                                                                                                                                                          |
    |------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad.html#torchjd.aggregation.UPGrad) (recommended) | [UPGradWeighting](https://torchjd.org/stable/docs/aggregation/upgrad#torchjd.aggregation.UPGradWeighting)              | [Jacobian Descent For Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232)                                                                                |
    | [AlignedMTL](https://torchjd.org/stable/docs/aggregation/aligned_mtl#torchjd.aggregation.AlignedMTL)       | [AlignedMTLWeighting](https://torchjd.org/stable/docs/aggregation/aligned_mtl#torchjd.aggregation.AlignedMTLWeighting) | [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/pdf/2305.19000)                                                                          |
    | [CAGrad](https://torchjd.org/stable/docs/aggregation/cagrad#torchjd.aggregation.CAGrad)                    | [CAGradWeighting](https://torchjd.org/stable/docs/aggregation/cagrad#torchjd.aggregation.CAGradWeighting)              | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048)                                                                         |
    | [ConFIG](https://torchjd.org/stable/docs/aggregation/config#torchjd.aggregation.ConFIG)                    | -                                                                                                                      | [ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks](https://arxiv.org/pdf/2408.11104)                                                       |
    | [Constant](https://torchjd.org/stable/docs/aggregation/constant#torchjd.aggregation.Constant)              | [ConstantWeighting](https://torchjd.org/stable/docs/aggregation/constant#torchjd.aggregation.ConstantWeighting)        | -                                                                                                                                                                    |
    | [DualProj](https://torchjd.org/stable/docs/aggregation/dualproj#torchjd.aggregation.DualProj)              | [DualProjWeighting](https://torchjd.org/stable/docs/aggregation/dualproj#torchjd.aggregation.DualProjWeighting)        | [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840)                                                                                  |
    | [GradDrop](https://torchjd.org/stable/docs/aggregation/graddrop#torchjd.aggregation.GradDrop)              | -                                                                                                                      | [Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808)                                                    |
    | [IMTLG](https://torchjd.org/stable/docs/aggregation/imtl_g#torchjd.aggregation.IMTLG)                      | [IMTLGWeighting](https://torchjd.org/stable/docs/aggregation/imtl_g#torchjd.aggregation.IMTLGWeighting)                | [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/)                                                                             |
    | [Krum](https://torchjd.org/stable/docs/aggregation/krum#torchjd.aggregation.Krum)                          | [KrumWeighting](https://torchjd.org/stable/docs/aggregation/krum#torchjd.aggregation.KrumWeighting)                    | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)  |
    | [Mean](https://torchjd.org/stable/docs/aggregation/mean#torchjd.aggregation.Mean)                          | [MeanWeighting](https://torchjd.org/stable/docs/aggregation/mean#torchjd.aggregation.MeanWeighting)                    | -                                                                                                                                                                    |
    | [MGDA](https://torchjd.org/stable/docs/aggregation/mgda#torchjd.aggregation.MGDA)                          | [MGDAWeighting](https://torchjd.org/stable/docs/aggregation/mgda#torchjd.aggregation.MGDAWeighting)                    | [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S1631073X12000738)                    |
    | [NashMTL](https://torchjd.org/stable/docs/aggregation/nash_mtl#torchjd.aggregation.NashMTL)                | -                                                                                                                      | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017)                                                                                         |
    | [PCGrad](https://torchjd.org/stable/docs/aggregation/pcgrad#torchjd.aggregation.PCGrad)                    | [PCGradWeighting](https://torchjd.org/stable/docs/aggregation/pcgrad#torchjd.aggregation.PCGradWeighting)              | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782)                                                                                         |
    | [Random](https://torchjd.org/stable/docs/aggregation/random#torchjd.aggregation.Random)                    | [RandomWeighting](https://torchjd.org/stable/docs/aggregation/random#torchjd.aggregation.RandomWeighting)              | [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603)                                              |
    | [Sum](https://torchjd.org/stable/docs/aggregation/sum#torchjd.aggregation.Sum)                             | [SumWeighting](https://torchjd.org/stable/docs/aggregation/sum#torchjd.aggregation.SumWeighting)                       | -                                                                                                                                                                    |
    | [Trimmed Mean](https://torchjd.org/stable/docs/aggregation/trimmed_mean#torchjd.aggregation.TrimmedMean)   | -                                                                                                                      | [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf)                                      |

    ## Contribution
    Please read the [Contribution page](CONTRIBUTING.md).

    ## Citation
    If you use TorchJD for your research, please cite:
    ```
    @article{jacobian_descent,
    title={Jacobian Descent For Multi-Objective Optimization},
    author={Quinton, Pierre and Rey, Valérian},
    journal={arXiv preprint arXiv:2406.16232},
    year={2024}
    }
    ```

    END of the README.md file.
    END of the system prompt.
    """

    agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=SYSTEM_PROMPT,
        tools=tools,
    )
    response = await agent.ainvoke(
        {
            # "messages": "is there any java code in torchjd?"
            # "messages": "can you remove TorchJD/torchjd from the vector store?"
            "messages": "can you query the vector store for 'class UPGrad('"
        }
    )
    print_messages(response)


if __name__ == "__main__":
    asyncio.run(main())
