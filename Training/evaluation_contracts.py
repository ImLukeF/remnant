"""Participant exclusion survives model changes and partially reused development corpora."""


def participant_registry(selection,plan):
    known=set(selection["previouslyObservedSubjects"]+selection["trainingSubjects"]+selection["validationSubjects"])
    recorded=set(plan["previouslyObservedSubjects"])
    held_out=set(plan["testSubjects"])
    if (not plan["fittingForbidden"] or not known.issubset(recorded) or held_out&recorded
            or len(held_out)!=len(plan["testSubjects"])):
        raise ValueError("Independent test participants must exclude every previously observed person.")
    return sorted(recorded|held_out)
