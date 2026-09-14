import React, { useState, useEffect } from 'react';
import { Task, Priority, Area } from '../types';

interface TaskModalProps {
  task: Task | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (task: Task) => void;
  areas: Area[];
}

const TaskModal: React.FC<TaskModalProps> = ({ task, isOpen, onClose, onSave, areas }) => {
  const [formData, setFormData] = useState<Partial<Task>>({});

  useEffect(() => {
    if (task) {
        setFormData({ ...task });
    }
  }, [task]);

  if (!isOpen || !task) return null;

  const handleSave = () => {
      if (formData.title) {
          onSave(formData as Task);
          onClose();
      }
  };

  const handleChange = (field: keyof Task, value: any) => {
      setFormData(prev => ({ ...prev, [field]: value }));
  };

  const currentAreaColor = areas.find(a => a.id === formData.area_id)?.color || 'bg-slate-500';

  return (
    <div className="fixed inset-0 bg-background-dark/30 dark:bg-black/50 backdrop-blur-sm z-40 transition-opacity flex items-center justify-center p-4">
      <div className="relative w-full max-w-3xl bg-white dark:bg-slate-900 rounded-xl shadow-2xl z-50 flex flex-col max-h-[90vh] overflow-hidden border border-slate-200 dark:border-slate-800 animate-fade-in-up">
        {/* Header */}
        <div className="flex items-start justify-between p-6 pb-2 border-b border-transparent">
          <div className="flex items-start gap-3 w-full mr-4">
            <button 
                onClick={() => handleChange('completed', !formData.completed)}
                className={`mt-1.5 flex-shrink-0 w-6 h-6 border-2 rounded-lg flex items-center justify-center transition-colors focus:outline-none ${formData.completed ? 'bg-primary border-primary' : 'border-slate-300 dark:border-slate-600 hover:border-primary'}`}
            >
                {formData.completed && <span className="material-icons text-sm text-white">check</span>}
            </button>
            <div className="w-full">
              <input 
                className="w-full text-2xl font-bold bg-transparent border-none p-0 focus:ring-0 text-slate-900 dark:text-white placeholder-slate-400 outline-none" 
                value={formData.title || ''}
                onChange={(e) => handleChange('title', e.target.value)}
                placeholder="Task title"
              />
              <div className="flex items-center gap-2 mt-1 text-sm text-slate-500 dark:text-slate-400">
                <span>in</span>
                <div className="relative group flex items-center gap-1">
                   <span className={`w-2 h-2 rounded-full ${currentAreaColor}`}></span>
                   <select 
                      value={formData.area_id || ''} 
                      onChange={(e) => handleChange('area_id', parseInt(e.target.value, 10))}
                      className="bg-transparent border-none text-sm font-medium focus:ring-0 cursor-pointer p-0"
                   >
                       {areas.map(area => (
                           <option key={area.id} value={area.id}>{area.name}</option>
                       ))}
                   </select>
                </div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={onClose} className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
              <span className="material-icons">close</span>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 pt-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                {/* Due Date */}
                <div className="flex items-center gap-3">
                    <div className="w-8 flex justify-center text-slate-400"><span className="material-icons">event</span></div>
                    <div className="flex-1">
                        <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">Due Date</label>
                        <input 
                            type="date"
                            className="bg-transparent border-none p-0 text-sm font-medium text-slate-700 dark:text-slate-200 focus:ring-0"
                            value={formData.due_date ? formData.due_date.split('T')[0] : ''}
                            onChange={(e) => handleChange('due_date', e.target.value ? new Date(e.target.value).toISOString() : undefined)}
                        />
                    </div>
                </div>
                {/* Priority */}
                <div className="flex items-center gap-3">
                    <div className="w-8 flex justify-center text-slate-400"><span className="material-icons">flag</span></div>
                    <div className="flex-1">
                        <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">Priority</label>
                        <select 
                             value={formData.priority || ''}
                             onChange={(e) => handleChange('priority', e.target.value)}
                             className="bg-transparent border-none p-0 text-sm font-medium text-slate-700 dark:text-slate-200 focus:ring-0 w-full"
                        >
                            <option value="">None</option>
                            <option value={Priority.Low}>Low</option>
                            <option value={Priority.Medium}>Medium</option>
                            <option value={Priority.High}>High</option>
                        </select>
                    </div>
                </div>
            </div>

            <div className="mb-8">
                <div className="flex items-center gap-2 mb-2">
                    <span className="material-icons text-slate-400 text-sm">description</span>
                    <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Description</h3>
                </div>
                <textarea 
                    className="w-full min-h-[120px] p-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800/50 rounded-lg border-transparent focus:border-primary focus:ring-0 resize-none outline-none" 
                    value={formData.description || ''}
                    onChange={(e) => handleChange('description', e.target.value)}
                    placeholder="Add a detailed description..."
                ></textarea>
            </div>
        </div>
        
        {/* Footer */}
        <div className="p-4 bg-slate-50 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 flex items-center justify-end gap-3">
             <button onClick={handleSave} className="px-5 py-2 text-sm font-medium text-white bg-primary hover:bg-primary-hover rounded-lg shadow-lg shadow-primary/30 transition-all">
                Save Changes
            </button>
        </div>
      </div>
    </div>
  );
};

export default TaskModal;
