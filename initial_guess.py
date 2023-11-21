from audio_transformations import extract_context_with_silent_gap,extract_truth_context,process_audio_files,audio_to_tf_parts,process_batch,process_single_sample,process_to_tf_representation,plot_spectrum,reshape_data,stft_to_audio,fill_gap_with_repeated_contexts


def fill_gap_with_repeated_contexts(X_train_tf, sample_index, gap_start, gap_end, context_size=14, plotting = True):
    """
    Fill the gap in the STFT representation by repeating the last five bins of the left context
    and the first five bins of the right context over their respective halves of the gap.
    """
    gap_length = gap_end - gap_start + 1
    stft_data= X_train_tf[sample_index]
    # Extract the relevant context bins
    left_context = stft_data[:, gap_start - context_size:gap_start, :]
    right_context = stft_data[:, gap_end + 1:gap_end + 1 + context_size, :]

    # Repeat the last five bins of the left context for the first half of the gap
    for i in range(gap_length // 2):
        stft_data[:, gap_start + i, :] = left_context[:, i % context_size, :]

    # Repeat the first five bins of the right context for the second half of the gap
    second_half_start = gap_start + gap_length // 2
    for i in range(second_half_start, gap_end + 1):
        stft_data[:, i, :] = right_context[:, (i - second_half_start) % context_size, :]
    # Extract the filled-in gap with 28 time bins
    filled_gap = stft_data[:, 29:57, :]  # Move the gap to the right by one time bin
    # Replace the original gap with the filled-in gap in X_train_tf
    X_train_tf[sample_index, :,28:56, :] = filled_gap  # Move the gap to the right by one time bin

    if (plotting == True):
      #Plot the filled gap with the repeated context approach
      print("Filled Gap (Repeated Contexts):")
      plot_spectrum(stft_data[:, 28:56, 0], stft_data[:, 28:56, 1], hop_length, fs_train)
      # Plot for the first gap and its context using real and imaginary parts separately
      print("Whole Spectrum with Filled Gap (Repeated Contexts):")
      plot_spectrum(X_train_tf[sample_index, :, :, 0], X_train_tf[sample_index, :, :, 1], hop_length, fs_train)
