for i in range(rounds):
    # m = max(floor(c*n), 1)
    # m = n
    # sample_clients = np.random.choice(n, m)
    # if i > 0.75*rounds and federatedLoc :
    #     federatedLoc = False
    tmp_glob = None
    tmp_loc = None
    tmp_loc_count = None
    for client in range(n):
        glob, loc, performance = clients[client].train(theta_glob, theta_loc)

        print(f"client - {client}, round - {i}/{rounds}")

        train_performance[client, :, i*local_epochs *
                          num_batch:(i+1)*local_epochs*num_batch] = copy.deepcopy(performance)

        performance = clients[client].test()
        test_performance[client, :, i *
                         num_batch:(i+1)*num_batch] = copy.deepcopy(performance)


        if federatedGlob:
            if tmp_glob is None:
                tmp_glob = glob
            else:
                for k in tmp_glob.keys():
                    tmp_glob[k] += glob[k]

        if federatedLoc:      
            if tmp_loc is None:
                tmp_loc = loc
                tmp_loc_count = dict.fromkeys(loc.keys(),1)
            else:
                for k in loc.keys():
                    if k not in tmp_loc:
                        tmp_loc[k] = loc[k]
                        tmp_loc_count[k] = 1
                    else:
                        tmp_loc[k] += loc[k]
                        tmp_loc_count[k] +=1

    dsp.clear_output(wait=True)
    avg = np.average(train_performance[0, 4, i*local_epochs *
                                       num_batch:(i+1)*local_epochs*num_batch])
    print(f"Training Loss: {avg}")
    # Dynamic plotting
    plot_loss_train.append(avg)

    avg = np.average(test_performance[0, 4, i * num_batch:(i+1)*num_batch])
    plot_loss_test.append(avg)
    print(f"Test Loss: {avg}")

    plt.clf()
    plt.plot(plot_loss_train)
    plt.plot(plot_loss_test)
    dsp.display(plt.gcf())
    if federatedGlob:
        if theta_glob is None:
            theta_glob = {}
        for k in tmp_glob.keys():
            theta_glob[k] = torch.div(tmp_glob[k], n)
    if federatedLoc:     
        if theta_loc is None:
            theta_loc = {}
        for k in tmp_loc.keys():
            tmp_loc[k] = torch.div(tmp_loc[k], tmp_loc_count[k])

notify.send("Analysis complete")
