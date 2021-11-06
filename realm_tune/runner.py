def main():
    args = parse_arguments()
    alg = OptunaHyperparamTuner(args)
    best_trial_name = alg.run()
    config = alg.config

if __name__ == "__main__":
    main()
