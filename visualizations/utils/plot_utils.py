import datetime as dt

# =====================================
#          Social Network
# =====================================
class plot_utils(object):
    @staticmethod
    def features_for_prev_days(current_day, day_dict, feature_list, number_of_previous_days=30):
        # Get days where data exists for this uid for the given hour bin id
        before = [
            current_day + dt.timedelta(x)
            for x in range(-number_of_previous_days, 0)
            if current_day + dt.timedelta(x) in day_dict
        ]
        prev_features = {}
        for past_day in before:
            for feature_dict in day_dict[past_day].values():
                for feature_type in feature_list:
                    if feature_type not in feature_dict:
                        continue
                    mergin_feature = prev_features.setdefault(feature_type, {})

                    # Merge the feature dicts for past days into a single dict.
                    # E.g. call_in feature will be single entry in the dict.
                    for obj_id, events in feature_dict[feature_type].items():
                        mergin_feature.setdefault(obj_id, []).extend(events)
        return prev_features

    @staticmethod
    def cut_top_n(top_contacts_dict, n):
        new_dict = {}
        for feature_type, subject_dict in top_contacts_dict.items():
            sorted_tuples = sorted(subject_dict.items(),
                                key=lambda x: x[1], reverse=True)[:n]
            new_dict[feature_type] = [uid for uid, val in sorted_tuples]
        return new_dict

    @staticmethod
    def union_and_cut_top_n(top_contacts_dict, n):
        new_dict = {}
        for val_dict in top_contacts_dict.values():
            for k, v in val_dict.items():
                if k in new_dict:
                    new_dict[k] += v
                else:
                    new_dict[k] = v
        #print "Before sort and cut"
        #print sorted(new_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_tuples = sorted(new_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        #print "After sort and cut:"
        #print sorted_tuples
        #print "------------------------------------------------------------------"
        return [uid for uid, val in sorted_tuples]

    @staticmethod
    def top_contacts_helper(dur_or_freq, separate_in_out, current_day, day_dict, days_before_onset, n):
        top_contacts_dict = {}
        feature_type_list = ["call_in", "call_out"]
        past_feature_dict = plot_utils.features_for_prev_days(
            current_day, day_dict, feature_type_list, days_before_onset)
        # Populate the top contacts list based on previous days
        for feature_type in feature_type_list:
            top_contacts_feature_dict = {}
            for obj_id, events in past_feature_dict.get(feature_type, {}).items():
                if obj_id not in top_contacts_feature_dict:
                    top_contacts_feature_dict[obj_id] = 0
                if dur_or_freq == "frequency":
                    top_contacts_feature_dict[obj_id] += len(events)
                elif dur_or_freq == "duration":
                    for duration, _ in events:
                        top_contacts_feature_dict[obj_id] += duration
            top_contacts_dict[feature_type] = top_contacts_feature_dict

        if separate_in_out:
            return plot_utils.cut_top_n(top_contacts_dict, n)
        else:
            return plot_utils.union_and_cut_top_n(top_contacts_dict, n)

    @staticmethod
    def top_n_contacts_by_call_duration(separate_in_out, current_day, day_dict, days_before_onset=30, n=3):
        return plot_utils.top_contacts_helper("duration", separate_in_out, current_day, day_dict, days_before_onset, n)

    @staticmethod
    def top_n_contacts_by_call_frequency(separate_in_out, current_day, day_dict, days_before_onset=30, n=3):
        return plot_utils.top_contacts_helper("frequency", separate_in_out, current_day, day_dict, days_before_onset, n)

    @staticmethod
    def unseen_features_for_prev_days(day_dict, feature, current_day, bin_id, number_of_previous_days=30):
        """ Returns a list of all the features that are present in the current day but have not been seen the past 30 days (number_of_previous_days) 
            Example: locations visited today that have not be visited past 30 days. 
        """
        todays_features = day_dict[current_day][bin_id].get(feature, {}).keys()
        if len(todays_features) == 0:
            return []
        before = [
            current_day + dt.timedelta(x)
            for x in range(-number_of_previous_days, 0)
            if current_day + dt.timedelta(x) in day_dict
        ]
        prev_features = [
            elem 
            for past_day in before 
            for elem in day_dict[past_day].get(bin_id, {}).get(feature, {}).keys()
        ]
        unseen_features = [
            elem 
            for elem in todays_features 
            if elem not in prev_features
        ]
        return unseen_features
