import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel

from utils import load_eval_dataset, calculate_scores_per_section, EXAMPLE_COUNTS, SUBSET_MAPPING, _get_reward_model, RewardPipeline, check_tokenizer_chat_template


def main():
    parser = argparse.ArgumentParser(description="Evaluate a reward model.")
    parser.add_argument("--dataset", type=str, default="allenai/reward-bench", help="The dataset to evaluate on.")
    parser.add_argument("--model", type=str, required=True, help="The model to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size to use.")

    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="The chat template to use (defaults to from tokenizer, from chattemplate).",
    )

    args = parser.parse_args()

    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.chat_template:
        from fastchat.conversation import get_conv_template

        conv = get_conv_template(args.chat_template)
    else:
        conv = None

    ##############
    # load dataset
    ##############
    logger.info("*** Load dataset ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("Running core eval dataset.")

    # primary set compiles slightly more information
    dataset, subsets = load_eval_dataset(
        core_set=True,
        conv=conv,
        custom_dialogue_formatting=False,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "prompt"],
    )

    logger.info("*** Load reward model ***")

    ################################
    # Load classifier model pipeline
    ################################

    tokenizer.padding_side = "left"
    truncation = False
    reward_pipeline_kwargs = {
        "batch_size": args.batch_size,  # eval_args.inference_batch_size,
        "truncation": truncation,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    model_kwargs = {"device_map": "auto"}
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_reward_model(base_pretrained_class, base_class)

    model = cls_class.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
    )

    reward_pipe = RewardPipeline(
            model=model,
            tokenizer=tokenizer
        )
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = accelerator.prepare(reward_pipe.model)
    reward_pipe.model = model

    ###############
    # Run inference
    ###############

    results = []
    scores_chosen = []
    scores_rejected = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
        rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            score_chosen_batch = [result["score"] for result in rewards_chosen]
            score_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            score_chosen_batch = rewards_chosen.to(torch.float32).cpu().numpy().tolist()
            score_rejected_batch = rewards_rejected.to(torch.float32).cpu().numpy().tolist()

        # log results
        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
        ]
        scores_chosen.extend(score_chosen_batch)
        scores_rejected.extend(score_rejected_batch)

    ############################
    # compile scores
    ############################
    # calculate accuracy
    accuracy = sum(results) / len(results)
    logger.info(f"Results: {accuracy}, on {len(results)} prompts")

    # compute mean and std of scores, chosen and rejected, then margin between them
    logger.info(f"Mean chosen: {np.mean(scores_chosen)}, std: {np.std(scores_chosen)}")
    logger.info(f"Mean rejected: {np.mean(scores_rejected)}, std: {np.std(scores_rejected)}")
    logger.info(f"Mean margin: {np.mean(np.array(scores_chosen) - np.array(scores_rejected))}")

    if args.dataset == "allenai/reward-bench":
        out_dataset = dataset.add_column("results", results)
        if args.debug:
            subsets = subsets[:10]
        out_dataset = out_dataset.add_column("subsets", subsets)
        out_dataset = out_dataset.to_pandas()  # I know this is meh

        results_grouped = {}
        present_subsets = np.unique(out_dataset["subsets"])
        for subset in present_subsets:
            subset_dataset = out_dataset[out_dataset["subsets"] == subset]
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            logger.info(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

        results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        logger.info(f"Results: {results_section}")


    ############################
    # compile scores
    ############################
    # save score in json to args.output_dir + args.model + ".json"
    # output_path = args.output_dir + args.model + ".json"
    # dirname = os.path.dirname(output_path)
    # os.makedirs(dirname, exist_ok=True)

    # # remove old data
    # if os.path.exists(output_path):
    #     os.remove(output_path)

    # with open(output_path, "w") as f:
    #     json.dump(
    #         {
    #             "accuracy": accuracy,
    #             "num_prompts": len(results),
    #             "model": args.model,
    #             "ref_model": args.ref_model,
    #             "tokenizer": args.model,
    #             "chat_template": args.chat_template,
    #             "extra_results": results_grouped if args.dataset == "allenai/reward-bench" else None,
    #         },
    #         f,
    #     )

    # # if save_all is passed, save a large jsonl with all scores_chosen, scores_rejected
    # if args.save_all:
    #     output_path = args.output_dir + args.model + "_all.jsonl"
    #     dirname = os.path.dirname(output_path)
    #     os.makedirs(dirname, exist_ok=True)

    #     # remove old data
    #     if os.path.exists(output_path):
    #         os.remove(output_path)

    #     with open(output_path, "w") as f:
    #         for chosen, rejected in zip(scores_chosen, scores_rejected):
    #             f.write(json.dumps({"chosen": scores_chosen, "rejected": scores_rejected}) + "\n")

if __name__ == "__main__":
    main()